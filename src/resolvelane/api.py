"""HTTP application entry point."""

import logging
import os
import time
from datetime import datetime, timezone
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel

from resolvelane import metrics

from resolvelane.access import Principal, Role, TokenRegistry, load_token_registry
from resolvelane.case_store import CaseStore, load_case_store
from resolvelane.knowledge import HelpCenter
from resolvelane.llm import load_llm_provider, load_prices
from resolvelane.memory import load_summary_queue
from resolvelane.payments import PaymentProvider, load_payment_dispatcher
from resolvelane.recognition import IssueRecognizer, load_local_embedder, load_recognizer
from resolvelane.reply_agent import ReplyAgent, load_reply_agent
from resolvelane.domain.case import Case, CaseStatus
from resolvelane.domain.refund import PaymentStatus, RefundStatus
from resolvelane.evaluation.reports import REGRESSION_TOLERANCE, REPORT_ID, ReportStore, load_report_store
from resolvelane.refunds import RefundInvalid, RefundProposal, payment_state
from resolvelane.schemas import (CaseListView, CaseView, ChatRequest, ChatResponse, CitationView, CustomerOrderView,
                                 CustomerRefundView, ErrorResponse, EvalReportListView,
                                 EvalReportView, OrderListView, PrincipalView, QueuePageView, RefundActionResponse,
                                 RefundDecisionRequest, RefundProposalRequest, RefundRejectionRequest,
                                 RefundRevisionRequest, ResolveRequest, SupportActionResponse, SupportCaseView,
                                 SupportReplyRequest)
from resolvelane.services import support_desk
from resolvelane.services.order_lookup import OrderAccess, OrderSource, lookup_order
from resolvelane.services.support_chat import (CaseNotFound, IdempotencyKeyReused, OrderMismatch, get_case,
                                               handle_message)
from resolvelane.simulator.fixtures import ORDERS, POLICIES
from resolvelane.skills import load_skills
from resolvelane.tools import ToolGateway
from resolvelane.simulator.help_center import ARTICLES
from resolvelane.simulator.order_source import InMemoryOrderSource
from resolvelane.simulator.payment_provider import load_payment_provider

logger = logging.getLogger("resolvelane.api")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


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


def require_support_agent(principal: Annotated[Principal, Depends(require_principal)]) -> Principal:
    if principal.role is not Role.SUPPORT_AGENT:
        raise HTTPException(403, "This endpoint is available to support agents only")
    return principal


def require_operator(principal: Annotated[Principal, Depends(require_principal)]) -> Principal:
    if principal.role is not Role.OPERATOR:
        raise HTTPException(403, "This endpoint is available to operators only")
    return principal


IdempotencyKey = Annotated[str | None, Header(
    alias="Idempotency-Key", min_length=1, max_length=128, pattern=r"^[\x21-\x7e]+$",
    description="A unique value per request, such as a UUID. Retrying with the same key and body "
                "returns the original response instead of applying the request again.")]


def _customer_refunds(case: Case) -> list[CustomerRefundView]:
    # The provider's details stay internal; a support agent explains a delayed refund.
    views = []
    for refund in case.refunds:
        if refund.status is not RefundStatus.APPROVED:
            continue
        sent = refund.payment is not None and refund.payment.status is PaymentStatus.SUCCEEDED
        views.append(CustomerRefundView(
            refund_id=refund.refund_id, status=refund.status, currency=refund.currency,
            amount_minor=refund.decided.amount_minor, lines=list(refund.decided.lines),
            delivery_fee_minor=refund.decided.delivery_fee_minor, approved_at=refund.decided_at,
            payment_status=payment_state(refund), sent_at=refund.payment.completed_at if sent else None))
    return views


def _proposal(body: RefundProposalRequest) -> RefundProposal:
    return RefundProposal(tuple((item.item_id, item.quantity) for item in body.items), body.include_delivery_fee,
                          body.reason)


def create_app(token_registry: TokenRegistry | None = None, order_source: OrderSource | None = None,
               case_store: CaseStore | None = None, recognizer: IssueRecognizer | None = None,
               reply_agent: ReplyAgent | None = None, payment_provider: PaymentProvider | None = None,
               evaluation_reports: ReportStore | None = None) -> FastAPI:
    owned_store = load_case_store() if case_store is None else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        if app.state.payments is not None:
            app.state.payments.close()
        if app.state.mcp is not None:
            app.state.mcp.close()
        if app.state.summaries is not None:
            app.state.summaries.close()
        if owned_store is not None:  # a store passed in belongs to the caller
            owned_store.close()

    app = FastAPI(title="ResolveLane API", version="0.1.0", lifespan=lifespan)
    app.state.token_registry = token_registry if token_registry is not None else load_token_registry()
    app.state.order_source = order_source if order_source is not None else InMemoryOrderSource(ORDERS)
    app.state.case_store = case_store if case_store is not None else owned_store
    # Recognition and reply writing share one provider, so they share its circuit breaker too.
    # Recognition and help center search share the local embedding model, loaded once.
    load_prices()  # costs in metrics and reply traces; checked at start-up
    llm = load_llm_provider() if recognizer is None or reply_agent is None else None
    local_embedder = load_local_embedder() if recognizer is None or reply_agent is None else None
    app.state.recognizer = (recognizer if recognizer is not None
                            else load_recognizer(llm=llm, local_embedder=local_embedder))
    # External services' tools (MCP) are only used by the reply agent.
    app.state.mcp = _load_mcp() if reply_agent is None and llm is not None else None
    app.state.reply_agent = (reply_agent if reply_agent is not None
                             else load_reply_agent(llm, POLICIES, HelpCenter(ARTICLES, local_embedder),
                                                   load_skills(), ToolGateway(external=app.state.mcp)))
    # Summaries are only needed by the reply agent, and use the same provider.
    app.state.summaries = load_summary_queue(getattr(app.state.reply_agent, "llm", None), app.state.case_store)
    app.state.payment_provider = payment_provider if payment_provider is not None else load_payment_provider()
    app.state.payments = load_payment_dispatcher(app.state.case_store, app.state.payment_provider)
    app.state.evaluation_reports = evaluation_reports if evaluation_reports is not None else load_report_store()

    @app.middleware("http")
    async def measure(request: Request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            # The route template, never the path itself: paths contain case and order IDs.
            route = getattr(request.scope.get("route"), "path", "unmatched")
            metrics.HTTP_REQUESTS.labels(route, request.method, str(status)).inc()
            metrics.HTTP_SECONDS.labels(route).observe(time.perf_counter() - started)

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/metrics", tags=["Health"], summary="Operational metrics in the Prometheus text format",
             response_class=Response, responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}})
    def read_metrics(principal: Annotated[Principal, Depends(require_operator)], request: Request) -> Response:
        store = request.app.state.case_store
        return Response(metrics.exposition(store, store.clock()), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/v1/me", response_model=PrincipalView, tags=["Access"],
             summary="Who the bearer token belongs to", responses={401: {"model": ErrorResponse},
                                                                   503: {"model": ErrorResponse}})
    def me(principal: Annotated[Principal, Depends(require_principal)]) -> PrincipalView:
        return PrincipalView(subject_id=principal.subject_id, role=principal.role.value)

    @app.get("/api/v1/orders", response_model=OrderListView, tags=["Orders"],
             summary="List the caller's simulated orders, newest first",
             responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    def list_orders(principal: Annotated[Principal, Depends(require_customer)], request: Request) -> OrderListView:
        orders = request.app.state.order_source.for_customer(principal.subject_id)
        return OrderListView(orders=[CustomerOrderView.model_validate(order) for order in orders])

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
        idempotency_key: IdempotencyKey = None,
    ) -> ChatResponse:
        try:
            turn = handle_message(request.app.state.case_store, request.app.state.order_source,
                                  request.app.state.recognizer, principal.subject_id, body.message,
                                  body.case_id, body.order_id, request.app.state.reply_agent, idempotency_key,
                                  request.app.state.summaries)
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

    @app.get("/api/v1/cases", response_model=CaseListView, tags=["Cases"],
             summary="List the caller's cases, newest first",
             responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    def list_cases(
        principal: Annotated[Principal, Depends(require_customer)],
        request: Request,
        limit: Annotated[int, Query(ge=1, le=50)] = 20,
    ) -> CaseListView:
        cases = request.app.state.case_store.earlier_cases(principal.subject_id, EPOCH, limit)
        return CaseListView(cases=cases)

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
        return CaseView.model_validate({**vars(case), "refunds": _customer_refunds(case)})

    # --- support desk ------------------------------------------------------------------

    agent_errors = {401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}

    @app.get("/api/v1/support/cases", response_model=QueuePageView, tags=["Support desk"],
             summary="List cases in a status, longest waiting first", responses=agent_errors)
    def support_queue(
        principal: Annotated[Principal, Depends(require_support_agent)],
        request: Request,
        status: CaseStatus = CaseStatus.NEEDS_REVIEW,
        limit: Annotated[int, Query(ge=1, le=support_desk.MAX_PAGE)] = 20,
        cursor: Annotated[str | None, Query(max_length=512)] = None,
    ) -> QueuePageView:
        try:
            page = support_desk.queue(request.app.state.case_store, status, limit, cursor)
        except support_desk.InvalidCursor:
            raise HTTPException(400, "Invalid cursor") from None
        return QueuePageView(cases=page.entries, next_cursor=page.next_cursor)

    @app.get("/api/v1/support/cases/{case_id}", response_model=SupportCaseView, tags=["Support desk"],
             summary="Read a case with its internal details",
             responses={**agent_errors, 404: {"model": ErrorResponse}})
    def support_case(
        case_id: Annotated[str, Path(min_length=1, max_length=64)],
        principal: Annotated[Principal, Depends(require_support_agent)],
        request: Request,
    ) -> SupportCaseView:
        try:
            case, summary = support_desk.read_case(request.app.state.case_store, case_id)
        except support_desk.CaseNotFound:
            raise HTTPException(404, "Case not found") from None
        # The agent works the case, so they see its order; the order endpoints stay customer-only.
        order = request.app.state.order_source.get(case.order_id) if case.order_id else None
        return SupportCaseView.model_validate({**vars(case), "waiting_since": case.waiting_since,
                                               "summary": summary.text if summary else None,
                                               "order": CustomerOrderView.model_validate(order) if order else None})

    def _run(action, request: Request, response: Response, *args):
        try:
            result = action(request.app.state.case_store, *args)
        except support_desk.CaseNotFound:
            raise HTTPException(404, "Case not found") from None
        except support_desk.RefundNotFound:
            raise HTTPException(404, "Refund not found") from None
        except support_desk.CaseResolved:
            raise HTTPException(409, "This case is resolved") from None
        except support_desk.RefundConflict as error:
            raise HTTPException(409, str(error)) from None
        except RefundInvalid as error:
            raise HTTPException(422, str(error)) from None
        except support_desk.IdempotencyKeyReused:
            raise HTTPException(422, "This Idempotency-Key was already used for a different request") from None
        if result.replayed:
            response.headers["Idempotent-Replayed"] = "true"
        return result

    def _desk_action(action, request: Request, response: Response, *args) -> SupportActionResponse:
        result = _run(action, request, response, *args)
        return SupportActionResponse(case_id=result.case_id, status=result.status, message_count=result.message_count)

    def _refund_action(action, request: Request, response: Response, *args) -> RefundActionResponse:
        result = _run(action, request, response, *args)
        return RefundActionResponse(refund_id=result.refund_id, case_id=result.case_id, status=result.status,
                                    version=result.version, amount_minor=result.amount_minor,
                                    currency=result.currency)

    action_errors = {**agent_errors, 404: {"model": ErrorResponse},
                     409: {"model": ErrorResponse,
                           "description": "The case is resolved, or a refund proposal is still undecided"}}
    refund_errors = {**agent_errors, 404: {"model": ErrorResponse},
                     409: {"model": ErrorResponse, "description": "The refund or case no longer allows this action, "
                                                                  "e.g. the version reviewed is not the current one"},
                     422: {"description": "Invalid request, a refund the order does not allow, or a reused "
                                          "Idempotency-Key"}}

    @app.post("/api/v1/support/cases/{case_id}/messages", response_model=SupportActionResponse,
              tags=["Support desk"], summary="Reply to the customer as a support agent", responses=action_errors)
    def support_reply(
        case_id: Annotated[str, Path(min_length=1, max_length=64)],
        body: SupportReplyRequest,
        principal: Annotated[Principal, Depends(require_support_agent)],
        request: Request,
        response: Response,
        idempotency_key: IdempotencyKey = None,
    ) -> SupportActionResponse:
        return _desk_action(support_desk.reply, request, response, principal.subject_id, case_id, body.text,
                            idempotency_key)

    @app.post("/api/v1/support/cases/{case_id}/resolution", response_model=SupportActionResponse,
              tags=["Support desk"], summary="Resolve the case", responses=action_errors)
    def support_resolve(
        case_id: Annotated[str, Path(min_length=1, max_length=64)],
        body: ResolveRequest,
        principal: Annotated[Principal, Depends(require_support_agent)],
        request: Request,
        response: Response,
        idempotency_key: IdempotencyKey = None,
    ) -> SupportActionResponse:
        return _desk_action(support_desk.resolve, request, response, principal.subject_id, case_id, body.note,
                            idempotency_key)

    @app.post("/api/v1/support/cases/{case_id}/refunds", response_model=RefundActionResponse,
              tags=["Support desk"], summary="Propose a refund, priced from the order record",
              responses=refund_errors)
    def support_propose_refund(
        case_id: Annotated[str, Path(min_length=1, max_length=64)],
        body: RefundProposalRequest,
        principal: Annotated[Principal, Depends(require_support_agent)],
        request: Request,
        response: Response,
        idempotency_key: IdempotencyKey = None,
    ) -> RefundActionResponse:
        return _refund_action(support_desk.propose_refund, request, response, request.app.state.order_source,
                              principal.subject_id, case_id, _proposal(body), idempotency_key)

    @app.post("/api/v1/support/refunds/{refund_id}/versions", response_model=RefundActionResponse,
              tags=["Support desk"], summary="Revise a refund proposal with a new version",
              responses=refund_errors)
    def support_revise_refund(
        refund_id: Annotated[str, Path(min_length=1, max_length=64)],
        body: RefundRevisionRequest,
        principal: Annotated[Principal, Depends(require_support_agent)],
        request: Request,
        response: Response,
        idempotency_key: IdempotencyKey = None,
    ) -> RefundActionResponse:
        return _refund_action(support_desk.revise_refund, request, response, request.app.state.order_source,
                              principal.subject_id, refund_id, body.revises, _proposal(body), idempotency_key)

    @app.post("/api/v1/support/refunds/{refund_id}/approval", response_model=RefundActionResponse,
              tags=["Support desk"], summary="Approve the current version of a refund proposal",
              responses=refund_errors)
    def support_approve_refund(
        refund_id: Annotated[str, Path(min_length=1, max_length=64)],
        body: RefundDecisionRequest,
        principal: Annotated[Principal, Depends(require_support_agent)],
        request: Request,
        response: Response,
        idempotency_key: IdempotencyKey = None,
    ) -> RefundActionResponse:
        result = _refund_action(support_desk.approve_refund, request, response, request.app.state.order_source,
                                principal.subject_id, refund_id, body.version, body.note, idempotency_key)
        if request.app.state.payments is not None:
            request.app.state.payments.poke()  # send the payment now rather than at the next poll
        return result

    @app.post("/api/v1/support/refunds/{refund_id}/rejection", response_model=RefundActionResponse,
              tags=["Support desk"], summary="Reject the current version of a refund proposal",
              responses=refund_errors)
    def support_reject_refund(
        refund_id: Annotated[str, Path(min_length=1, max_length=64)],
        body: RefundRejectionRequest,
        principal: Annotated[Principal, Depends(require_support_agent)],
        request: Request,
        response: Response,
        idempotency_key: IdempotencyKey = None,
    ) -> RefundActionResponse:
        return _refund_action(support_desk.reject_refund, request, response, principal.subject_id, refund_id,
                              body.version, body.note, idempotency_key)

    @app.get("/api/v1/evaluation/reports", response_model=EvalReportListView, tags=["Evaluation"],
             summary="Saved evaluation runs, newest first, and the baseline",
             responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}})
    def list_evaluation_reports(principal: Annotated[Principal, Depends(require_operator)],
                                request: Request) -> EvalReportListView:
        store = request.app.state.evaluation_reports
        runs, unreadable = store.runs()
        return EvalReportListView(reports=runs, baseline=store.baseline(), tolerance=REGRESSION_TOLERANCE,
                                  unreadable=unreadable)

    @app.get("/api/v1/evaluation/reports/{report_id}", response_model=EvalReportView, tags=["Evaluation"],
             summary="One evaluation run with every reply, its checks and the judge's reasons",
             responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse},
                        404: {"model": ErrorResponse}})
    def read_evaluation_report(report_id: Annotated[str, Path(pattern=REPORT_ID)],
                               principal: Annotated[Principal, Depends(require_operator)],
                               request: Request) -> EvalReportView:
        found = request.app.state.evaluation_reports.get(report_id)
        if found is None:
            raise HTTPException(404, "Evaluation report not found")
        return found

    return app


def _load_mcp():
    if not os.environ.get("RESOLVELANE_MCP_CONFIG", "").strip():
        return None
    from resolvelane.mcp_client import load_mcp_hub  # needs the optional mcp package
    return load_mcp_hub()


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
