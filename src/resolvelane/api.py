"""HTTP application entry point."""

import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Request
from pydantic import BaseModel

from resolvelane.access import Principal, Role, TokenRegistry, load_token_registry
from resolvelane.assistant import respond
from resolvelane.schemas import ChatRequest, ChatResponse, CustomerOrderView, ErrorResponse
from resolvelane.services.order_lookup import OrderAccess, OrderSource, lookup_order
from resolvelane.simulator.fixtures import ORDERS
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


def create_app(token_registry: TokenRegistry | None = None, order_source: OrderSource | None = None) -> FastAPI:
    app = FastAPI(title="ResolveLane API", version="0.1.0")
    app.state.token_registry = token_registry if token_registry is not None else load_token_registry()
    app.state.order_source = order_source if order_source is not None else InMemoryOrderSource(ORDERS)

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
        summary="Send a support message about one of the caller's simulated orders",
        responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def chat(
        body: ChatRequest,
        principal: Annotated[Principal, Depends(require_customer)],
        request: Request,
    ) -> ChatResponse:
        order = None
        if body.order_id is not None:
            # Another customer's order yields None, the same as an unknown order.
            order = lookup_order(request.app.state.order_source, principal.subject_id, body.order_id).order
        answer = respond(body.message, body.order_id, order)
        return ChatResponse(reply=answer.reply, issues=answer.issues, order_id=body.order_id,
                            needs_human_review=answer.needs_human_review)

    return app


app = create_app()
