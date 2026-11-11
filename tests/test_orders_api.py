import asyncio
import json
import logging

import httpx
import pytest

from resolvelane.access import AuthConfigError, TokenRegistry, load_token_registry
from resolvelane.api import create_app

# Invented test credentials for fictional accounts.
CUSTOMER_101 = "test-token-customer-101-aaaaaaaa"
CUSTOMER_102 = "test-token-customer-102-bbbbbbbb"
SUPPORT_AGENT = "test-token-support-agent-cccccccc"

TOKENS = {
    CUSTOMER_101: {"subject_id": "demo-customer-101", "role": "customer"},
    CUSTOMER_102: {"subject_id": "demo-customer-102", "role": "customer"},
    SUPPORT_AGENT: {"subject_id": "demo-agent-301", "role": "support_agent"},
}


class Client:
    """Synchronous wrapper around httpx's in-process ASGI transport."""

    def __init__(self, app):
        self.app = app

    def get(self, path, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get(path, **kwargs)
        return asyncio.run(send())


@pytest.fixture
def client():
    return Client(create_app(token_registry=TokenRegistry(TOKENS)))


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_owner_reads_order_without_internal_ids(client):
    response = client.get("/api/v1/orders/demo-order-1001", headers=auth(CUSTOMER_101))
    assert response.status_code == 200
    body = response.json()
    assert body["order_id"] == "demo-order-1001"
    assert body["status"] == "delivered"
    assert body["total_minor"] == 2721
    assert [item["item_id"] for item in body["items"]] == ["item-1", "item-2"]
    assert body["delivery_events"][-1]["event_type"] == "delivered"
    assert "customer_id" not in body
    assert "merchant_id" not in body


def test_another_customers_order_looks_identical_to_an_unknown_order(client):
    denied = client.get("/api/v1/orders/demo-order-1001", headers=auth(CUSTOMER_102))
    unknown = client.get("/api/v1/orders/demo-order-9999", headers=auth(CUSTOMER_102))
    assert denied.status_code == unknown.status_code == 404
    assert denied.content == unknown.content
    assert denied.json() == {"detail": "Order not found"}


def test_denied_and_unknown_lookups_are_distinguished_in_logs(client, caplog):
    with caplog.at_level(logging.INFO, logger="resolvelane.api"):
        client.get("/api/v1/orders/demo-order-1001", headers=auth(CUSTOMER_102))
        client.get("/api/v1/orders/demo-order-9999", headers=auth(CUSTOMER_102))
    messages = [record.getMessage() for record in caplog.records]
    assert any("access=denied" in message and "demo-order-1001" in message for message in messages)
    assert any("access=not_found" in message and "demo-order-9999" in message for message in messages)


def test_client_supplied_identity_is_ignored(client):
    response = client.get(
        "/api/v1/orders/demo-order-1001",
        params={"customer_id": "demo-customer-101", "subject_id": "demo-customer-101"},
        headers={**auth(CUSTOMER_102), "X-Customer-Id": "demo-customer-101"},
    )
    assert response.status_code == 404


@pytest.mark.parametrize("headers", [
    {},
    {"Authorization": "Bearer "},
    {"Authorization": "Bearer not-a-configured-token-xxxxxx"},
    {"Authorization": f"Basic {CUSTOMER_101}"},
    {"Authorization": CUSTOMER_101},
])
def test_missing_or_invalid_credentials_are_rejected(client, headers):
    response = client.get("/api/v1/orders/demo-order-1001", headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_support_agent_cannot_use_the_customer_endpoint(client):
    response = client.get("/api/v1/orders/demo-order-1001", headers=auth(SUPPORT_AGENT))
    assert response.status_code == 403


def test_overlong_order_id_is_rejected(client):
    response = client.get("/api/v1/orders/" + "x" * 65, headers=auth(CUSTOMER_101))
    assert response.status_code == 422


def test_order_endpoint_is_unavailable_without_token_configuration():
    client = Client(create_app(token_registry=TokenRegistry({})))
    response = client.get("/api/v1/orders/demo-order-1001", headers=auth(CUSTOMER_101))
    assert response.status_code == 503
    assert client.get("/health").status_code == 200


def test_production_requires_token_configuration():
    with pytest.raises(AuthConfigError, match="production"):
        load_token_registry({"RESOLVELANE_ENV": "production"})


def test_development_without_tokens_loads_an_empty_registry():
    assert not load_token_registry({}).configured


def test_registry_is_loaded_from_environment():
    registry = load_token_registry({"RESOLVELANE_API_TOKENS": json.dumps(TOKENS)})
    principal = registry.authenticate(CUSTOMER_101)
    assert principal is not None
    assert principal.subject_id == "demo-customer-101"
    assert registry.authenticate("unknown-token") is None


@pytest.mark.parametrize("raw,message", [
    ("not json", "JSON object"),
    ("[]", "non-empty JSON object"),
    ("{}", "non-empty JSON object"),
    (json.dumps({"short": {"subject_id": "demo-customer-101", "role": "customer"}}), "at least"),
    (json.dumps({CUSTOMER_101: "demo-customer-101"}), "map to an object"),
    (json.dumps({CUSTOMER_101: {"subject_id": "", "role": "customer"}}), "subject_id"),
    (json.dumps({CUSTOMER_101: {"subject_id": "demo-customer-101", "role": "admin"}}), "known role"),
])
def test_invalid_token_configuration_is_rejected(raw, message):
    with pytest.raises(AuthConfigError, match=message):
        load_token_registry({"RESOLVELANE_API_TOKENS": raw})
