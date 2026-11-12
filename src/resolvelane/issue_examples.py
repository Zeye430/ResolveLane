"""Invented example messages for each issue, used by the embedding tiers of issue recognition.

UNKNOWN holds messages that raise no issue (greetings, thanks, questions about the support
agent), so such messages are matched confidently instead of being forced onto an issue.
Thresholds in embeddings.CALIBRATED_THRESHOLDS were measured with these examples; changing
them means running scripts/calibrate_embeddings.py again.
"""

from resolvelane.domain.case import Issue

ISSUE_EXAMPLES: dict[Issue, tuple[str, ...]] = {
    Issue.LATE_DELIVERY: (
        "My order arrived late", "The delivery took way too long", "It came an hour after the estimate",
        "My food is really late", "The courier was very slow", "It got here much later than promised",
        "I've been waiting ages for my delivery", "The order is delayed",
    ),
    Issue.MISSING_ITEM: (
        "An item is missing from my order", "My order never arrived", "I didn't receive my drink",
        "Part of my order is not in the bag", "They left out my side dish", "I only got half of what I paid for",
        "Nothing was delivered", "My dessert wasn't included",
    ),
    Issue.WRONG_ITEM: (
        "I got the wrong item", "This is not what I ordered", "They sent me someone else's food",
        "I received a different dish", "The order is mixed up", "I asked for chicken but got beef",
        "These aren't my items", "Wrong order delivered",
    ),
    Issue.DAMAGED_ITEM: (
        "My food arrived damaged", "The soup spilled everywhere", "The box was crushed",
        "The packaging was torn open", "The drink leaked all over the bag", "My pizza was squashed",
        "Everything was a mess when it arrived", "The container was broken",
    ),
    Issue.CANCELLATION: (
        "Why was my order cancelled?", "My order got cancelled", "Who cancelled my order?",
        "I want to cancel my order", "Can I still cancel?", "The app says my order was cancelled",
        "Please cancel the order", "My order was cancelled without warning",
    ),
    Issue.DELIVERY_STATUS: (
        "Where is my order?", "When will my food arrive?", "Can you track my order?",
        "How long until it gets here?", "Is my order on the way?", "What's the status of my delivery?",
        "Has the driver picked it up yet?", "Any update on my order?",
    ),
    Issue.UNKNOWN: (
        "Thanks!", "Thank you so much", "Hello", "Hi there", "Okay", "Great, that's all",
        "When will someone get back to me?", "Can I talk to a person?", "Are you a bot?", "Bye",
    ),
}
