"""Measure similarity thresholds for an embedding tier of issue recognition.

Each held-out message below is labeled by hand and does not appear in issue_examples.py. The
script reports the threshold and margin that answer the most messages correctly while keeping
wrong answers at or below 5% of messages; everything else is left to the next tier.

    python scripts/calibrate_embeddings.py local [model]   # e.g. BAAI/bge-small-en-v1.5
    python scripts/calibrate_embeddings.py remote          # uses RESOLVELANE_EMBEDDING_* settings
    add -v to list every message

The thresholds are fitted to these messages, so the reported accuracy is optimistic. Put the
result in embeddings.CALIBRATED_THRESHOLDS and re-run whenever the examples change.
"""

import os
import sys
from pathlib import Path

from resolvelane.domain.case import Issue
from resolvelane.embeddings import EmbeddingMatcher, LocalEmbedder, RemoteEmbedder
from resolvelane.recognition import DEFAULT_EMBEDDING_BASE_URL, DEFAULT_EMBEDDING_MODEL, DEFAULT_LOCAL_EMBEDDING_MODEL

L, M, W, D, C, S, N = (Issue.LATE_DELIVERY, Issue.MISSING_ITEM, Issue.WRONG_ITEM, Issue.DAMAGED_ITEM,
                       Issue.CANCELLATION, Issue.DELIVERY_STATUS, Issue.UNKNOWN)
HELD_OUT = [
    ("Still waiting, it's been forever", L), ("It showed up 40 minutes past the time you gave", L),
    ("Why did it take so long?", L), ("My lunch came way behind schedule", L), ("The rider took over an hour", L),
    ("It was supposed to be here at 7 and came at 8", L),
    ("It never came.", M), ("They forgot my fries", M), ("My drink wasn't in the bag", M), ("I'm short one burger", M),
    ("Where are my sauces? None were included", M), ("I paid for three wraps and got two", M),
    ("Got someone else's bag", W), ("This isn't what I ordered, I got a burger", W),
    ("I ordered vegetarian and it has meat", W), ("You gave me a large instead of a small", W),
    ("These are not my meals", W), ("The salad is a totally different one", W),
    ("The driver dropped it, everything is smashed", D), ("Half my food was all over the bag", D),
    ("The lid came off and coffee is everywhere", D), ("My cake is ruined", D), ("The bag was ripped and soaked", D),
    ("Noodles leaked through the box", D),
    ("Did you cancel it?", C), ("I got a message my order was called off", C),
    ("Stop my order please, I don't want it anymore", C), ("Why would the restaurant cancel?", C),
    ("Can I call off this order?", C), ("My order disappeared and says cancelled", C),
    ("Where's my food", S), ("Is it close?", S), ("How far away is the driver?", S), ("ETA?", S),
    ("Has it left the restaurant?", S), ("When does it get here", S),
    ("Thanks a lot!", N), ("Hi", N), ("ok cool", N), ("Is anyone there?", N), ("I'd like to speak to an agent", N),
    ("Appreciate it", N), ("Good evening", N), ("How long until a human replies?", N),
]
MAX_WRONG_SHARE = 0.05


def embedder_from_args(args):
    if args and args[0] == "local":
        model = args[1] if len(args) > 1 else DEFAULT_LOCAL_EMBEDDING_MODEL
        cache = os.environ.get("RESOLVELANE_MODEL_CACHE_DIR") or str(Path.home() / ".cache" / "resolvelane" / "models")
        return LocalEmbedder(model, cache)
    if args and args[0] == "remote":
        return RemoteEmbedder(os.environ.get("RESOLVELANE_EMBEDDING_BASE_URL") or DEFAULT_EMBEDDING_BASE_URL,
                              os.environ["RESOLVELANE_EMBEDDING_API_KEY"],
                              os.environ.get("RESOLVELANE_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL)
    sys.exit(__doc__)


def main(argv):
    verbose = "-v" in argv
    embedder = embedder_from_args([arg for arg in argv if arg != "-v"])
    matcher = EmbeddingMatcher(embedder, threshold=0, margin=0)
    vectors = embedder.embed([text for text, _ in HELD_OUT])
    results = [(text, truth, *matcher.best_match(vector)) for (text, truth), vector in zip(HELD_OUT, vectors)]

    best = None
    for threshold in (step / 100 for step in range(0, 100)):
        for margin in (step / 100 for step in range(0, 21)):
            answered = [r for r in results if r[3] >= threshold and r[4] >= margin]
            correct = sum(r[1] is r[2] for r in answered)
            wrong = len(answered) - correct
            # Most correct answers; on a tie, the stricter setting.
            key = (correct, -wrong, margin, threshold)
            if wrong <= MAX_WRONG_SHARE * len(results) and (best is None or key > best[0]):
                best = (key, threshold, margin, correct, wrong)
    _, threshold, margin, correct, wrong = best
    top1 = sum(r[1] is r[2] for r in results)
    print(f"{embedder.name}: closest issue right for {top1}/{len(results)} messages")
    print(f"threshold {threshold:.2f}, margin {margin:.2f}: correct {correct}, wrong {wrong}, "
          f"left to the next tier {len(results) - correct - wrong}")
    if verbose:
        for text, truth, got, score, lead in results:
            answered = score >= threshold and lead >= margin
            mark = "pass" if not answered else ("ok" if truth is got else "WRONG")
            print(f"  {mark:5} truth={truth.value:15} closest={got.value:15} score={score:.2f} lead={lead:.2f} | {text}")


if __name__ == "__main__":
    main(sys.argv[1:])
