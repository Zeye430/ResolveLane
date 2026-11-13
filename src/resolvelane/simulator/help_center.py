"""Invented help center articles for the simulated marketplace.

They describe no real company's terms. One article has an expired earlier version with
different numbers, so tests can show that outdated guidance is never used.
"""

from resolvelane.domain.knowledge import HelpArticle


def _article(article_id, version, title, effective_from, sections, effective_until=None):
    return HelpArticle.model_validate({
        "article_id": article_id, "version": version, "title": title, "effective_from": effective_from,
        "effective_until": effective_until,
        "sections": [{"section_id": sid, "heading": heading, "text": text} for sid, heading, text in sections],
    })


ARTICLES = (
    _article("refund-timing", 1, "How refunds work", "2025-01-01T00:00:00Z", [
        ("arrival", "When a refund arrives",
         "Approved refunds reach the original payment method within 7 to 10 business days."),
    ], effective_until="2026-01-01T00:00:00Z"),
    _article("refund-timing", 2, "How refunds work", "2026-01-01T00:00:00Z", [
        ("decision", "Who decides on a refund",
         "Refunds are never issued automatically. A support agent reviews each request against the order "
         "record and decides whether a full or partial refund applies."),
        ("arrival", "When a refund arrives",
         "Once a support agent approves a refund, it reaches the original payment method within 3 to 5 "
         "business days. Some banks take longer to show it on a statement."),
        ("partial", "Partial refunds",
         "When only some items were affected, a refund usually covers those items rather than the whole order. "
         "Delivery fees are refunded only when the whole order was affected."),
    ]),
    _article("cancel-order", 1, "Cancelling an order", "2026-01-01T00:00:00Z", [
        ("before-accept", "Before the restaurant accepts",
         "You can cancel an order in the app until the restaurant accepts it. Cancelling at this point is free "
         "and the full amount is returned to your payment method."),
        ("after-accept", "After the restaurant accepts",
         "Once the restaurant has started preparing your food, the app no longer offers cancellation. Ask "
         "support in the chat; a support agent reviews the request because the food may already be made."),
        ("cancelled-by-restaurant", "If the restaurant cancels",
         "If a restaurant cancels your order, you are not charged. The app shows the cancellation in your "
         "order history."),
    ]),
    _article("change-address", 1, "Changing the delivery address", "2026-01-01T00:00:00Z", [
        ("before-courier", "Before a courier is assigned",
         "You can edit the delivery address in the order screen until a courier is assigned to the order."),
        ("after-courier", "After a courier is assigned",
         "Once a courier is assigned, the address can no longer be changed in the app, because the route has "
         "already been planned. If the address is wrong, ask support in the chat."),
    ]),
    _article("report-problem", 1, "Reporting a problem with your food", "2026-01-01T00:00:00Z", [
        ("time-limit", "When to report",
         "Report missing, wrong or damaged items within 24 hours of delivery, in the chat for that order."),
        ("photos", "Photos help",
         "For wrong or damaged items, a photo of the food and the packaging helps the support agent check "
         "the report against the order."),
        ("what-happens", "What happens next",
         "A support agent compares your report with the order record and decides on the outcome. You will "
         "see their reply in the same conversation."),
    ]),
    _article("late-orders", 1, "Late deliveries", "2026-01-01T00:00:00Z", [
        ("estimates", "Delivery estimates",
         "The estimated delivery time is calculated when the restaurant accepts the order and can change with "
         "traffic and preparation time."),
        ("compensation", "Compensation for late orders",
         "Late orders are not compensated automatically. A support agent reviews late deliveries case by case."),
    ]),
    _article("talk-to-agent", 1, "Talking to a support agent", "2026-01-01T00:00:00Z", [
        ("how", "How to reach an agent",
         "Ask for a person in the chat. The assistant passes your case to a support agent, and their reply "
         "appears in the same conversation."),
    ]),
    _article("allergies", 1, "Allergies and dietary needs", "2026-01-01T00:00:00Z", [
        ("restaurant-info", "Where allergy information comes from",
         "Allergen and ingredient details are provided by each restaurant. Check the dish description and "
         "contact the restaurant before ordering if you have a severe allergy."),
    ]),
)
