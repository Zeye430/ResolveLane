"""HTTP application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Request, Response
from pydantic import BaseModel

from resolvelane.access import Principal, Role, TokenRegistry, load_token_registry
from resolvelane.case_store import CaseStore, load_case_store
from resolvelane.knowledge import HelpCenter
from resolvelane.llm import load_llm_provider
from resolvelane.recognition import IssueRecognizer, load_local_embedder, load_recognizer
from resolvelane.reply_agent import ReplyAgent, load_reply_agent
from resolvelane.schemas import CaseView, ChatRequest, ChatResponse, CitationView, CustomerOrderView, ErrorResponse
from resolvelane.services.order_lookup import OrderAccess, OrderSource, lookup_order
from resolvelane.services.support_chat import (CaseNotFound, IdempotencyKeyReused, OrderMismatch, get_case,
                                               handle_message)
from resolvelane.simulator.fixtures import ORDERS, POLICIES
from resolvelane.simulator.help_center import ARTICLES
from resolvelane.simulator.order_source import InMemoryOrderSource

logger = logging.getLogger("resolvelane.api")


class HealthResponse(BaseModel):
    status: str


def require_principal(request: Request) -> Principal:
    """Resolve the caller from the bearer token; client-supplied identity fields are never used."""
    registry: TokenRegistry = request.app.state.token_registry
    if not registry.configured:
        raise HTTPException(503, "Authentication is not configured")
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    principal = registry.authenticate(token) if scheme.lower() == "bearer" and token else None
    if principal is None:
        raise HTTPException(401, "A valid bearer token is required", headers={"WWW-Authenticate": "Bearer"})
    return principal


def require_customer(principal: Annotated[Principal, Depends(require_principal)]) -> Principal:
    if principal.role is not Role.CUSTOMER:
        raise HTTPException(403, "This endpoint is available to customers only")
    return principal


def create_app(token_registry: TokenRegistry | None = None, order_source: OrderSource | None = None,
               case_store: CaseStore | None = None, recognizer: IssueRecognizer | None = None,
               reply_agent: ReplyAgent | None = None) -> FastAPI:
    owned_store = load_case_store() if case_store is None else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        if owned_store is not None:  # a store passed in belongs to the caller
            owned_store.close()

    app = FastAPI(title="ResolveLane API", version="0.1.0", lifespan=lifespan)
    app.state.token_registry = token_registry if token_registry is not None else load_token_registry()
    app.state.order_source = order_source if order_source is not None else InMemoryOrderSource(ORDERS)
    app.state.case_store = case_store if case_store is not None else owned_store
    # Recognition and reply writing share one provider, so they share its circuit breaker too.
    # Recognition and help center search share the local embedding model, loaded once.
    llm = load_llm_provider() if recognizer is None or reply_agent is None else None
    local_embedder = load_local_embedder() if recognizer is None or reply_agent is None else None
    app.state.recognizer = (recognizer if recognizer is not None
                            else load_recognizer(llm=llm, local_embedder=local_embedder))
    app.state.reply_agent = (reply_agent if reply_agent is not None
                             else load_reply_agent(llm, POLICIES, HelpCenter(ARTICLES, local_embedder)))

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get(
        "/api/v1/orders/{order_id}",
        response_model=CustomerOrderView,
        tags=["Orders"],
        summary="Read one of the caller's simulated orders",
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse, "description": "The order does not exist or belongs to another customer"},
            503: {"model": ErrorResponse},
        },
    )
    def get_order(
        order_id: Annotated[str, Path(min_length=1, max_length=64)],
        principal: Annotated[Principal, Depends(require_customer)],
        request: Request,
    ) -> CustomerOrderView:
        result = lookup_order(request.app.state.order_source, principal.subject_id, order_id)
        logger.info("order_lookup access=%s subject=%s order_id=%r",
                    result.access.value, principal.subject_id, order_id)
        if result.customer_visible_access is not OrderAccess.GRANTED:
            raise HTTPException(404, "Order not found")
        return CustomerOrderView.model_validate(result.order)

    @app.post(
        "/api/v1/chat",
        response_model=ChatResponse,
        tags=["Chat"],
        summary="Send a support message, starting a new case or continuing an existing one",
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse, "description": "The case does not exist or belongs to another customer"},
            409: {"model": ErrorResponse, "description": "The case is about a different order"},
            422: {"description": "Invalid request, or an Idempotency-Key already used for a different request"},
            503: {"model": ErrorResponse},
        },
    )
    def chat(
        body: ChatRequest,
        principal: Annotated[Principal, Depends(require_customer)],
        request: Request,
        response: Response,
        idempotency_key: Annotated[str | None, Header(
            alias="Idempotency-Key", min_length=1, max_length=128, pattern=r"^[\x21-\x7e]+$",
            description="A unique value per message, such as a UUID. Retrying with the same key and body "
                        "returns the original response instead of sending the message again.")] = None,
    ) -> ChatResponse:
        try:
            turn = handle_message(request.app.state.case_store, request.app.state.order_source,
                                  request.app.state.recognizer, principal.subject_id, body.message,
                                  body.case_id, body.order_id, request.app.state.reply_agent, idempotency_key)
        except CaseNotFound:
            raise HTTPException(404, "Case not found") from None
        except OrderMismatch as error:
            raise HTTPException(409, str(error)) from None
        except IdempotencyKeyReused:
            raise HTTPException(422, "This Idempotency-Key was already used for a different request") from None
        if turn.replayed:
            response.headers["Idempotent-Replayed"] = "true"
        return ChatResponse(case_id=turn.case_id, case_status=turn.case_status, reply=turn.reply.reply,
                            issues=turn.reply.issues, order_id=turn.order_id,
                            needs_human_review=turn.reply.needs_human_review,
                            citations=[CitationView.model_validate(citation) for citation in turn.reply.citations])

    @app.get(
        "/api/v1/cases/{case_id}",
        response_model=CaseView,
        tags=["Cases"],
        summary="Read one of the caller's cases with its conversation",
        responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse},
                   404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def read_case(
        case_id: Annotated[str, Path(min_length=1, max_length=64)],
        principal: Annotated[Principal, Depends(require_customer)],
        request: Request,
    ) -> CaseView:
        try:
            case = get_case(request.app.state.case_store, principal.subject_id, case_id)
        except CaseNotFound:
            raise HTTPException(404, "Case not found") from None
        return CaseView.model_validate(case)

    return app


def _show_application_logs() -> None:
    """Print ResolveLane's INFO records when served by uvicorn, which only configures its own loggers."""
    package_logger = logging.getLogger("resolvelane")
    if not package_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s %(message)s"))
        package_logger.addHandler(handler)
        package_logger.setLevel(logging.INFO)


_show_application_logs()  # before the app is built, so startup messages are shown
app = create_app()
