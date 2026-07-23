#!/usr/bin/env python3
"""One-time batch backfill: problem-level sub_issue tags for historical GP reviews.

Fisher asked (Jul 13 & 17) for a problem-level drill-down of the ~9.3k/month
actioned bad-review increment — head-vs-long-tail (Pareto) and long-term-stable-
vs-version-burst. Today's data is classified only by reply-template `stage2_topic`
("which template to send"), not by "what problem the user actually has". This
script produces the historical sub_issue layer so Fisher sees a FULL Pareto/burst
on day one, not just reviews tagged going forward.

WHAT IT DOES
  1. Reads the topic-tagged GP reviews from BigQuery
     (`review_events_analytics.review_events_raw_latest`, the CDC/raw view whose
     fields live in a JSON `data` STRING column — extracted via JSON_VALUE).
  2. Assigns each review a `sub_issue` (from main.SUB_ISSUES_BY_TOPIC, keyed by
     the review's STORED topic — the topic is the source of truth, we do NOT
     re-derive it) and a rule-derived `solvable_type`:
       - topics whose only slot is SENTIMENT (恶意/无缘由/Suggestions) and any
         single-slot topic  -> assigned deterministically, NO LLM call.
       - multi-slot topics (Device, Account, Gacha, ...) -> Gemini picks the slot.
  3. Writes {event_id, game, stage2_topic, stage2_issue_type, sub_issue,
     solvable_type, ...} to a NEW BQ table `review_events_analytics.review_sub_issues`
     (NOT back into the 158k-doc Firestore collection — keeps it untouched and
     avoids write cost/risk). The /sub-issue-analytics endpoint joins this table
     with the raw view.

IDEMPOTENT: event_ids already present in review_sub_issues are skipped, so the
job can be re-run / resumed safely.

REPRODUCIBLE: emits exact counts (rows read, LLM vs rule, per-topic coverage) so
ZR can defend every number to Fisher. Nothing here is a benchmark — it is the
live historical set.

USAGE
  # dry run over a small sample (classifies, prints, writes nothing):
  python backfill_sub_issues.py --limit 200 --dry-run
  # full backfill:
  python backfill_sub_issues.py
  # resume (idempotent — skips already-tagged):
  python backfill_sub_issues.py

ENV
  GOOGLE_CLOUD_PROJECT   GCP project (default: hoyoverseguojihua)
  BQ_DATASET             dataset (default: review_events_analytics)
  BQ_SOURCE_VIEW         source view (default: review_events_raw_latest)
  BQ_TARGET_TABLE        target table (default: review_sub_issues)
  BACKFILL_MAX_WORKERS   LLM concurrency (default: 8)
"""

import argparse
import concurrent.futures
import datetime
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# Reuse the live pipeline's constants + helpers so backfilled sub_issues are
# defined identically to the ones tagged live (same enum, same solvable_type
# rule, same model, same JSON-call plumbing). This is what makes the historical
# and going-forward Pareto directly comparable.
import main  # noqa: E402
from google.cloud import bigquery  # noqa: E402
from google.genai import types  # noqa: E402


PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT") or main._project_id() or "hoyoverseguojihua"
DATASET = os.getenv("BQ_DATASET", "review_events_analytics")
SOURCE_VIEW = os.getenv("BQ_SOURCE_VIEW", "review_events_raw_latest")
TARGET_TABLE = os.getenv("BQ_TARGET_TABLE", "review_sub_issues")
MAX_WORKERS = int(os.getenv("BACKFILL_MAX_WORKERS", "8"))

TARGET_REF = f"{PROJECT}.{DATASET}.{TARGET_TABLE}"
SOURCE_REF = f"{PROJECT}.{DATASET}.{SOURCE_VIEW}"

# gemini-3.1-flash-lite published price (per 1M tokens, Vertex). Used ONLY for a
# rough cost ESTIMATE in the summary — labeled as an estimate, never a bill.
_PRICE_IN_PER_1M = 0.10
_PRICE_OUT_PER_1M = 0.40

_TARGET_SCHEMA = [
    bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("game", "STRING"),
    bigquery.SchemaField("stage2_topic", "STRING"),
    bigquery.SchemaField("stage2_issue_type", "STRING"),
    bigquery.SchemaField("sub_issue", "STRING"),
    bigquery.SchemaField("solvable_type", "STRING"),
    bigquery.SchemaField("confidence", "FLOAT"),
    # "rule" (deterministic), "rule_empty_body", or the model name (LLM-classified).
    bigquery.SchemaField("classified_by", "STRING"),
    bigquery.SchemaField("source", "STRING"),  # always "backfill" here
    bigquery.SchemaField("tagged_at", "TIMESTAMP"),
]


def _log(msg: str) -> None:
    print(f"[backfill] {msg}", flush=True)


def _ensure_target_table(client: bigquery.Client) -> None:
    """Create review_sub_issues if it doesn't exist (clustered on event_id for
    the join, and on game/topic for the per-topic coverage cuts)."""
    try:
        client.get_table(TARGET_REF)
        _log(f"target table exists: {TARGET_REF}")
        return
    except Exception:
        pass
    table = bigquery.Table(TARGET_REF, schema=_TARGET_SCHEMA)
    table.clustering_fields = ["event_id", "game", "stage2_topic"]
    client.create_table(table)
    _log(f"created target table: {TARGET_REF}")


def _load_already_tagged(client: bigquery.Client) -> set:
    """event_ids already in review_sub_issues (idempotency / resume)."""
    try:
        client.get_table(TARGET_REF)
    except Exception:
        return set()
    q = f"SELECT DISTINCT event_id FROM `{TARGET_REF}`"
    rows = client.query(q).result()
    tagged = {r["event_id"] for r in rows}
    _log(f"already-tagged event_ids: {len(tagged)}")
    return tagged


def _fetch_topic_tagged_reviews(client: bigquery.Client, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Stream the topic-tagged GP reviews from the raw view. Base filter matches
    the analytics source-of-truth cut (real CSC webhook rows, topic present)."""
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    q = f"""
    SELECT
      JSON_VALUE(data, "$.event_id")           AS event_id,
      JSON_VALUE(data, "$.game")               AS game,
      JSON_VALUE(data, "$.stage2_topic")       AS stage2_topic,
      JSON_VALUE(data, "$.stage2_issue_type")  AS stage2_issue_type,
      JSON_VALUE(data, "$.language")           AS language,
      JSON_VALUE(data, "$.review_body")        AS review_body
    FROM `{SOURCE_REF}`
    WHERE JSON_VALUE(data, "$.event_id") LIKE "evt_gp_%"
      AND JSON_VALUE(data, "$.stage2_topic") IS NOT NULL
    {limit_clause}
    """
    _log("querying topic-tagged reviews from BigQuery ...")
    rows = list(client.query(q).result())
    _log(f"fetched {len(rows)} topic-tagged rows")
    # Dedupe by event_id (the raw_latest view is already latest-per-doc, but be
    # defensive — a dupe would double-count in the Pareto).
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        eid = r["event_id"]
        if eid and eid not in out:
            out[eid] = dict(r)
    if len(out) != len(rows):
        _log(f"deduped to {len(out)} unique event_ids")
    return list(out.values())


# -----------------------------
# Classification
# -----------------------------
def _backfill_sub_issue_prompt(topic: str, slots: List[str], body: str, language: str) -> str:
    """Focused, topic-PINNED sub_issue prompt. The topic is already decided (it's
    the stored source-of-truth), so we only ask the model to pick the single most
    specific slot within it — cheaper than re-running full Stage 2 and it can't
    drift the topic. Same evidence-phrase / match-on-meaning discipline as live."""
    slots_str = ", ".join([f'"{s}"' for s in slots])
    return f"""You are labeling a Google Play review already classified under the topic "{topic}".
Pick the SINGLE most specific sub_issue — the concrete PROBLEM the user has — from
this topic's slots ONLY: [{slots_str}]

RULES:
- Reviews are MULTILINGUAL. Match on MEANING/CONCEPT, not the exact English word.
- Choose the slot whose problem the review text actually supports.
- If the review is emotion-only, or no slot clearly matches, use "SENTIMENT".
- Return ONLY the sub_issue; do NOT second-guess the topic.

Review:
language: {language}
body: {main._truncate(body, 800)}

Return JSON: {{"sub_issue": "one slot or SENTIMENT", "confidence": 0.0-1.0}}"""


def _backfill_schema(slots: List[str]) -> Dict[str, Any]:
    enum = list(slots)
    if main.SENTIMENT_SUB_ISSUE not in enum:
        enum.append(main.SENTIMENT_SUB_ISSUE)
    return {
        "type": "OBJECT",
        "properties": {
            "sub_issue": {"type": "STRING", "enum": enum},
            "confidence": {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["sub_issue"],
        "propertyOrdering": ["sub_issue", "confidence"],
    }


# Rough token accounting for the cost estimate (thread-safe).
_TOK_LOCK = threading.Lock()
_est_in_tokens = 0
_est_out_tokens = 0
_llm_calls = 0


def _classify_one(review: Dict[str, Any]) -> Dict[str, Any]:
    """Return a target-table row dict for one review. Deterministic (no LLM) for
    SENTIMENT-only / single-slot topics; LLM otherwise."""
    global _est_in_tokens, _est_out_tokens, _llm_calls
    topic = (review.get("stage2_topic") or "").strip()
    issue_type = (review.get("stage2_issue_type") or "").strip() or None
    body = (review.get("review_body") or "").strip()
    language = review.get("language") or "unknown"

    slots = main.SUB_ISSUES_BY_TOPIC.get(topic, [])
    sub_issue: Optional[str]
    confidence: Optional[float]
    classified_by: str

    # Deterministic short-circuits (the majority of the corpus): no LLM, no
    # variance, no cost. SENTIMENT-only topics (恶意/无缘由/Suggestions) and any
    # topic with exactly one slot have a fixed answer.
    non_sentiment = [s for s in slots if s != main.SENTIMENT_SUB_ISSUE]
    if not slots or slots == [main.SENTIMENT_SUB_ISSUE]:
        sub_issue, confidence, classified_by = main.SENTIMENT_SUB_ISSUE, 1.0, "rule"
    elif len(slots) == 1:
        sub_issue, confidence, classified_by = slots[0], 1.0, "rule"
    elif len(non_sentiment) == 1:
        # e.g. a topic mapping to [x, SENTIMENT] — still a two-way call; let the
        # LLM decide problem-vs-emotion. (No topic currently hits this, but keep
        # it correct if the enum grows.)
        sub_issue, confidence, classified_by = None, None, ""
    elif not body:
        # Multi-slot topic but no text to classify on — fall back to SENTIMENT so
        # the row is still counted, flagged so it's auditable.
        sub_issue, confidence, classified_by = main.SENTIMENT_SUB_ISSUE, 0.0, "rule_empty_body"
    else:
        sub_issue, confidence, classified_by = None, None, ""

    if classified_by == "":
        # LLM path.
        prompt = _backfill_sub_issue_prompt(topic, slots, body, language)
        schema = _backfill_schema(slots)
        try:
            r = main._gen_json(
                main.VERTEX_MODEL_STAGE2, prompt, max_tokens=256,
                response_schema=schema, retries=1,
            )
            sub_issue = (r.get("sub_issue") or main.SENTIMENT_SUB_ISSUE).strip()
            confidence = float(r.get("confidence") or 0.0)
            classified_by = main.VERTEX_MODEL_STAGE2
        except Exception as e:
            _log(f"LLM classify failed for {review.get('event_id')}: {e} -> SENTIMENT")
            sub_issue, confidence, classified_by = main.SENTIMENT_SUB_ISSUE, 0.0, "rule_llm_failed"
        with _TOK_LOCK:
            _llm_calls += 1
            _est_in_tokens += max(1, len(prompt) // 4)   # ~4 chars/token heuristic
            _est_out_tokens += 16
        # Guard: never trust an out-of-enum value.
        if sub_issue not in slots and sub_issue != main.SENTIMENT_SUB_ISSUE:
            sub_issue = main.SENTIMENT_SUB_ISSUE

    solvable_type = main._derive_solvable_type(topic, issue_type, sub_issue)
    return {
        "event_id": review["event_id"],
        "game": review.get("game"),
        "stage2_topic": topic,
        "stage2_issue_type": issue_type,
        "sub_issue": sub_issue,
        "solvable_type": solvable_type,
        "confidence": confidence,
        "classified_by": classified_by,
        "source": "backfill",
        "tagged_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def _write_rows(client: bigquery.Client, rows: List[Dict[str, Any]]) -> None:
    """Append a batch to review_sub_issues via a load job (no streaming buffer)."""
    if not rows:
        return
    job_config = bigquery.LoadJobConfig(
        schema=_TARGET_SCHEMA,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    job = client.load_table_from_json(rows, TARGET_REF, job_config=job_config)
    job.result()  # wait


def main_run() -> int:
    ap = argparse.ArgumentParser(description="Backfill sub_issue tags to BigQuery.")
    ap.add_argument("--limit", type=int, default=None, help="only process first N rows (sampling / test)")
    ap.add_argument("--dry-run", action="store_true", help="classify + summarize, write nothing")
    ap.add_argument("--batch-size", type=int, default=2000, help="rows per BQ load job")
    args = ap.parse_args()

    _log(f"project={PROJECT} source={SOURCE_REF} target={TARGET_REF} workers={MAX_WORKERS}")
    client = bigquery.Client(project=PROJECT)

    if not args.dry_run:
        _ensure_target_table(client)
    already = set() if args.dry_run else _load_already_tagged(client)

    reviews = _fetch_topic_tagged_reviews(client, args.limit)
    todo = [r for r in reviews if r["event_id"] not in already]
    _log(f"to classify: {len(todo)} (skipped {len(reviews) - len(todo)} already tagged)")
    if not todo:
        _log("nothing to do.")
        return 0

    # Split deterministic (no LLM) from LLM-needed so we can report the split and
    # only pay for the reviews that actually need a model call.
    def _needs_llm(r: Dict[str, Any]) -> bool:
        slots = main.SUB_ISSUES_BY_TOPIC.get((r.get("stage2_topic") or "").strip(), [])
        non_sentiment = [s for s in slots if s != main.SENTIMENT_SUB_ISSUE]
        return len(slots) >= 2 and len(non_sentiment) >= 1 and bool((r.get("review_body") or "").strip()) \
            and not (len(non_sentiment) == 1 and len(slots) == 1)

    llm_ct = sum(1 for r in todo if _needs_llm(r))
    _log(f"deterministic (rule): {len(todo) - llm_ct}  |  LLM: {llm_ct}")

    results: List[Dict[str, Any]] = [None] * len(todo)  # type: ignore
    start = time.time()

    # Concurrency helps the LLM-bound rows; deterministic ones are instant.
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_classify_one, r): i for i, r in enumerate(todo)}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            if done % 1000 == 0:
                _log(f"classified {done}/{len(todo)} ({done*100//len(todo)}%)")

    # Coverage + distribution summary (reproducible numbers for Fisher).
    from collections import Counter
    by_sub = Counter(r["sub_issue"] for r in results)
    by_solv = Counter(r["solvable_type"] for r in results)
    by_src = Counter(r["classified_by"] for r in results)
    _log(f"done classifying {len(results)} in {time.time()-start:.1f}s")
    _log(f"solvable_type: {dict(by_solv)}")
    _log(f"classified_by: {dict(by_src)}")
    _log(f"top sub_issues: {by_sub.most_common(15)}")

    est_cost = (_est_in_tokens / 1_000_000 * _PRICE_IN_PER_1M) + (_est_out_tokens / 1_000_000 * _PRICE_OUT_PER_1M)
    _log(f"LLM calls: {_llm_calls}  est_in_tok≈{_est_in_tokens}  est_out_tok≈{_est_out_tokens}  "
         f"est_cost≈${est_cost:.2f} (ESTIMATE, not a bill)")

    if args.dry_run:
        _log("dry-run: no rows written.")
        return 0

    _log(f"writing {len(results)} rows to {TARGET_REF} in batches of {args.batch_size} ...")
    for i in range(0, len(results), args.batch_size):
        batch = results[i:i + args.batch_size]
        _write_rows(client, batch)
        _log(f"wrote {min(i + args.batch_size, len(results))}/{len(results)}")

    total_tagged = len(already) + len(results)
    _log(f"BACKFILL COMPLETE. review_sub_issues now covers {total_tagged} event_ids.")
    return 0


if __name__ == "__main__":
    sys.exit(main_run())
