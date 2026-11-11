"""Core types for a simulated delivery marketplace."""

from datetime import datetime
from enum import Enum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class OrderStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class DeliveryEventType(str, Enum):
    ORDER_PLACED = "order_placed"
    MERCHANT_ACCEPTED = "merchant_accepted"
    COURIER_ASSIGNED = "courier_assigned"
    PICKED_UP = "picked_up"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class OrderItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    quantity: int = Field(gt=0, strict=True)
    unit_price_minor: int = Field(ge=0, strict=True)


class DeliveryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_type: DeliveryEventType
    occurred_at: AwareDatetime


class Order(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    order_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    merchant_id: str = Field(min_length=1)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    status: OrderStatus
    items: tuple[OrderItem, ...] = Field(min_length=1)
    delivery_fee_minor: int = Field(ge=0, strict=True)
    tax_minor: int = Field(ge=0, strict=True)
    total_minor: int = Field(ge=0, strict=True)
    estimated_delivery_at: AwareDatetime
    delivery_events: tuple[DeliveryEvent, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_order(self):
        item_ids = [item.item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Item IDs must be unique within an order")

        subtotal = sum(item.quantity * item.unit_price_minor for item in self.items)
        if self.total_minor != subtotal + self.delivery_fee_minor + self.tax_minor:
            raise ValueError("Order total must equal item subtotal plus delivery fee and tax")

        event_types = [event.event_type for event in self.delivery_events]
        if event_types[0] != DeliveryEventType.ORDER_PLACED:
            raise ValueError("The first delivery event must be order_placed")
        if len(event_types) != len(set(event_types)):
            raise ValueError("Delivery event types cannot be repeated")
        timestamps: list[datetime] = [event.occurred_at for event in self.delivery_events]
        if any(first > second for first, second in zip(timestamps, timestamps[1:])):
            raise ValueError("Delivery events must be in chronological order")
        if self.estimated_delivery_at < timestamps[0]:
            raise ValueError("Estimated delivery cannot be before the order was placed")

        terminal = {
            OrderStatus.DELIVERED: DeliveryEventType.DELIVERED,
            OrderStatus.CANCELLED: DeliveryEventType.CANCELLED,
        }
        expected_terminal = terminal.get(self.status)
        actual_terminals = {DeliveryEventType.DELIVERED, DeliveryEventType.CANCELLED} & set(event_types)
        if expected_terminal is None and actual_terminals:
            raise ValueError("An in-progress order cannot have a terminal event")
        if expected_terminal is not None and (
            actual_terminals != {expected_terminal} or event_types[-1] != expected_terminal
        ):
            raise ValueError("Order status must match its final delivery event")
        return self


class SupportPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    title: str = Field(min_length=1)
    issue_type: str = Field(min_length=1)
    effective_from: AwareDatetime
    effective_until: AwareDatetime | None = None
    guidance: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_effective_interval(self):
        if self.effective_until is not None and self.effective_until <= self.effective_from:
            raise ValueError("Policy end must be after its start")
        return self
