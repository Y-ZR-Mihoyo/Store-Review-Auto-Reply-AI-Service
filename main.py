import os
import csv
import io
import json
import math
import time
import logging
import datetime
import hashlib
import re
import secrets
import threading
from typing import Any, Dict, List, Optional, Tuple

import functions_framework
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest

from google import genai
from google.genai import types

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)


# -----------------------------
# Small utils
# -----------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)


def _utc_iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except (ValueError, OverflowError):
        return default


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _truncate(s: str, n: int = 800) -> str:
    s = s or ""
    return s if len(s) <= n else (s[:n] + "…<truncated>")


def _count_runes(s: str) -> int:
    return len(s or "")


# -----------------------------
# Config
# -----------------------------
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "")
WEBHOOK_HEADER = os.getenv("WEBHOOK_HEADER", "x-rpc-ai-review-event")

VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
# NOTE: gemini-2.5-flash has a known JSON truncation bug, using stable 2.0-flash-001 instead
VERTEX_MODEL_STAGE1 = os.getenv("VERTEX_MODEL_STAGE1", "gemini-2.0-flash-001")
VERTEX_MODEL_STAGE2 = os.getenv("VERTEX_MODEL_STAGE2", "gemini-2.0-flash-001")

DEFAULT_TEMPLATE_LANG = os.getenv("DEFAULT_TEMPLATE_LANG", "EN")
TEMPLATES_PATH = os.getenv("TEMPLATES_PATH", "templates.json")

BAD_RATING_MAX = _env_int("BAD_RATING_MAX", 2)
REQUIRES_TEXT = _env_bool("REQUIRES_TEXT", True)

STAGE1_RETRIES = _env_int("STAGE1_RETRIES", 2)
STAGE2_RETRIES = _env_int("STAGE2_RETRIES", 2)

# LLM Configuration
MAX_OUTPUT_TOKENS = 1024
STAGE2_CONFIDENCE_THRESHOLD = 0.7

# Firestore (REST) config
FIRESTORE_ENABLED = _env_bool("FIRESTORE_ENABLED", True)
FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE", "store-review-ai-replies").strip()
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "review_events").strip()
FIRESTORE_WRITE_TIMEOUT_SEC = _env_int("FIRESTORE_WRITE_TIMEOUT_SEC", 8)

# Async processing config
ASYNC_ENABLED = _env_bool("ASYNC_ENABLED", False)
CLOUD_TASKS_QUEUE = os.getenv("CLOUD_TASKS_QUEUE", "review-processing-queue")
CLOUD_TASKS_LOCATION = os.getenv("CLOUD_TASKS_LOCATION", "europe-west1")
CLOUD_TASKS_SERVICE_ACCOUNT = os.getenv("CLOUD_TASKS_SERVICE_ACCOUNT", "")
CLOUD_RUN_SERVICE_URL = os.getenv("CLOUD_RUN_SERVICE_URL", "")
INTERNAL_PROCESS_TOKEN = os.getenv("INTERNAL_PROCESS_TOKEN", "")

# CSC Callback config
CSC_CALLBACK_URL = os.getenv("CSC_CALLBACK_URL", "")
CSC_CALLBACK_TOKEN_HEADER = os.getenv("CSC_CALLBACK_TOKEN_HEADER", "x-rpc-ai-callback-token")
CSC_CALLBACK_TOKEN = os.getenv("CSC_CALLBACK_TOKEN", "")
CSC_CALLBACK_TIMEOUT_SEC = _env_int("CSC_CALLBACK_TIMEOUT_SEC", 10)

LANG_MAP = {
    # Full locale codes
    "en-us": "EN",
    "zh-cn": "CHS",
    "zh-tw": "CHT",
    "id-id": "ID",
    "de-de": "DE",
    "fr-fr": "FR",
    "es-es": "ES",
    "pt-pt": "PT",
    "ru-ru": "RU",
    "ko-kr": "KR",
    "vi-vn": "VN",
    # Google Play script subtag format
    "zh-hans": "CHS",
    "zh-hant": "CHT",
    # Bare language codes (Google Play sometimes sends these)
    "en": "EN",
    "zh": "CHS",
    "id": "ID",
    "de": "DE",
    "fr": "FR",
    "es": "ES",
    "pt": "PT",
    "ru": "RU",
    "ko": "KR",
    "vi": "VN",
}

GAME_BIZ_TO_GAME = {
    "googleplay_nap": "ZZZ",
    "googleplay_hk4e": "GI",
    "googleplay_hkrpg": "HSR",
}

JSON_HEADERS = {"Content-Type": "application/json"}

# EWMA Forecast Optimization config
EWMA_CONFIG_COLLECTION = "ewma_config"
EWMA_DAILY_COLLECTION = "ewma_daily_data"
EWMA_UPLOAD_LOG_COLLECTION = "ewma_upload_log"
EWMA_OPT_HISTORY_COLLECTION = "ewma_optimization_history"
EWMA_DEFAULT_ALPHA = 0.02157
EWMA_ALPHA_RANGE_START = 0.005
EWMA_ALPHA_RANGE_END = 0.30
EWMA_ALPHA_STEP = 0.005
EWMA_BURN_IN_DAYS = 20
EWMA_MIN_DAYS_OPTIMIZE = 90
EWMA_TRAIN_RATIO = 0.70
VALID_GAMES = {"GI", "HSR", "ZZZ"}


def _log_event(msg: str, data: Dict[str, Any]) -> None:
    logging.info(_json_dumps({"msg": msg, **data}))


# -----------------------------
# Template registry
# -----------------------------
TEMPLATE_REGISTRY: Dict[str, Any] = {}
TEMPLATE_INDEX: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
TOPICS_BY_GAME: Dict[str, List[str]] = {}
TOPIC_ISSUE_TYPE: Dict[Tuple[str, str], str] = {}


def _load_templates(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_template_index(registry: Dict[str, Any]) -> None:
    templates = registry.get("templates", [])
    idx: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    topics: Dict[str, set] = {}
    topic_it: Dict[Tuple[str, str], str] = {}

    for t in templates:
        game = (t.get("game") or "").strip()
        issue_type = (t.get("issue_type") or "").strip()
        topic = (t.get("topic") or "").strip()
        language = (t.get("language") or "").strip()
        template_id = (t.get("template_id") or "").strip()
        text = (t.get("template_text") or "")

        if not (game and issue_type and topic and language and template_id and text):
            continue

        idx[(game, issue_type, topic, language)] = t
        topics.setdefault(game, set()).add(topic)
        topic_it.setdefault((game, topic), issue_type)

    global TEMPLATE_INDEX, TOPICS_BY_GAME, TOPIC_ISSUE_TYPE
    TEMPLATE_INDEX = idx
    TOPICS_BY_GAME = {g: sorted(list(s)) for g, s in topics.items()}
    TOPIC_ISSUE_TYPE = topic_it


def _init_registry() -> None:
    global TEMPLATE_REGISTRY
    try:
        TEMPLATE_REGISTRY = _load_templates(TEMPLATES_PATH)
        _build_template_index(TEMPLATE_REGISTRY)
        _log_event(
            "template_registry_loaded",
            {
                "templates_path": TEMPLATES_PATH,
                "template_count": len(TEMPLATE_REGISTRY.get("templates", [])),
                "games": sorted(list(TOPICS_BY_GAME.keys())),
            },
        )
    except Exception as e:
        _log_event(
            "template_registry_load_failed",
            {"templates_path": TEMPLATES_PATH, "err": str(e)},
        )
        TEMPLATE_REGISTRY = {}
        TEMPLATE_INDEX.clear()
        TOPICS_BY_GAME.clear()


_init_registry()


# -----------------------------
# Vertex GenAI client (Gemini on Vertex)
# -----------------------------
def _project_id() -> str:
    return os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT") or os.getenv("PROJECT_ID") or ""


_GENAI_CLIENT: Optional[genai.Client] = None


def _get_genai_client() -> genai.Client:
    global _GENAI_CLIENT
    if _GENAI_CLIENT is not None:
        return _GENAI_CLIENT

    project = _project_id()
    if not project:
        raise RuntimeError("Missing project id (GOOGLE_CLOUD_PROJECT / PROJECT_ID not set).")

    _GENAI_CLIENT = genai.Client(vertexai=True, project=project, location=VERTEX_LOCATION)
    return _GENAI_CLIENT


# -----------------------------
# Cloud Tasks client (lazy)
# -----------------------------
_TASKS_CLIENT = None


def _get_tasks_client():
    global _TASKS_CLIENT
    if _TASKS_CLIENT is not None:
        return _TASKS_CLIENT
    from google.cloud import tasks_v2
    _TASKS_CLIENT = tasks_v2.CloudTasksClient()
    return _TASKS_CLIENT


# -----------------------------
# JSON parsing helpers
# -----------------------------
def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            return "\n".join(lines[1:-1]).strip()
        return text.strip("`").strip()
    return text


def _slice_first_json(text: str) -> str:
    text = _strip_code_fences(text)
    if not text:
        return text
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in ('{', '['):
            try:
                _, end = decoder.raw_decode(text, i)
                return text[i:end]
            except json.JSONDecodeError:
                continue
    return text


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output")
    return json.loads(_slice_first_json(text))


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# -----------------------------
# Result building helpers
# -----------------------------
def _build_result(
    event_id: str,
    order_id: str,
    action: str,
    payload: Dict[str, Any],
    gate_result: str = "ELIGIBLE",
    stage1: Optional[Dict[str, Any]] = None,
    stage2: Optional[Dict[str, Any]] = None,
    **extra
) -> Dict[str, Any]:
    """Build standardized result dictionary."""
    result = {
        "event_id": event_id,
        "order_id": str(order_id),
        "action": action,
    }
    
    for key in ["game_biz", "rating", "language", "store_type"]:
        if key in extra:
            result[key] = extra[key]
        elif key in payload:
            result[key] = payload.get(key)
    
    result["gate_result"] = gate_result
    
    if stage1 is not None:
        result["stage1"] = stage1
    if stage2 is not None:
        result["stage2"] = stage2
    
    result.update(extra)
    
    return result


def _respond(
    start_ms: int,
    event_id: str,
    order_id: str,
    action: str,
    payload: Dict[str, Any],
    gate_result: str = "ELIGIBLE",
    stage1: Optional[Dict[str, Any]] = None,
    stage2: Optional[Dict[str, Any]] = None,
    **extra
) -> Tuple[str, int, Dict[str, str]]:
    """Build result and finalize response in one call."""
    result = _build_result(
        event_id=event_id,
        order_id=order_id,
        action=action,
        payload=payload,
        gate_result=gate_result,
        stage1=stage1,
        stage2=stage2,
        **extra
    )
    latency = _now_ms() - start_ms
    _log_event("decision", {**result, "latency_ms": latency})
    _firestore_write_review_event_best_effort(
        event_id=event_id,
        payload=payload,
        result=result,
        latency_ms=latency,
    )
    return json.dumps(result, ensure_ascii=False), 200, JSON_HEADERS


def _validate_template_length(reply_text: str, max_chars: int) -> Optional[str]:
    """Validate template length. Returns error reason or None if valid."""
    if _count_runes(reply_text) > max_chars:
        return "template_too_long"
    return None


# -----------------------------
# GenAI call
# -----------------------------
def _extract_response_text(resp: Any) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Extract text from response, along with finish_reason and any safety info.
    Returns: (text, finish_reason, safety_info)
    """
    text = ""
    finish_reason = None
    safety_info = None

    try:
        candidates = getattr(resp, "candidates", None)
        if candidates and len(candidates) > 0:
            candidate = candidates[0]
            fr = getattr(candidate, "finish_reason", None)
            if fr is not None:
                finish_reason = str(fr)

            safety_ratings = getattr(candidate, "safety_ratings", None)
            if safety_ratings:
                blocked = [str(sr) for sr in safety_ratings if getattr(sr, "blocked", False)]
                if blocked:
                    safety_info = f"blocked_safety:{blocked}"

            content = getattr(candidate, "content", None)
            if content:
                parts = getattr(content, "parts", None)
                if parts and len(parts) > 0:
                    part = parts[0]
                    part_text = getattr(part, "text", None)
                    if part_text:
                        text = part_text
    except Exception as e:
        _log_event("extract_response_text_candidate_err", {"err": str(e)})

    if not text:
        try:
            resp_text = getattr(resp, "text", None)
            if resp_text:
                text = resp_text
        except Exception as e:
            _log_event("extract_response_text_fallback_err", {"err": str(e)})

    if not text:
        text = str(resp)

    return text, finish_reason, safety_info


def _gen_json(
    model: str,
    prompt: str,
    *,
    max_tokens: int = 1024,
    response_schema: Optional[Dict[str, Any]] = None,
    retries: int = 0,
) -> Dict[str, Any]:
    client = _get_genai_client()

    last_raw = ""
    last_err: Optional[Exception] = None
    last_finish_reason: Optional[str] = None

    for attempt in range(retries + 1):
        if attempt > 0:
            delay = min(0.5 * (2 ** (attempt - 1)), 2.0)
            time.sleep(delay)

        prompt_to_use = prompt if attempt == 0 else (
            "IMPORTANT: Return ONLY valid, complete JSON. No preamble, no markdown, no backticks, no comments. "
            "Ensure all strings are properly terminated and the JSON is complete.\n\n" + prompt
        )

        use_schema = response_schema if attempt == 0 else None

        try:
            cfg_obj = types.GenerateContentConfig(
                temperature=0,
                top_p=1,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                response_schema=use_schema,
            )
            cfg_to_use: Any = cfg_obj
        except Exception:
            cfg_dict: Dict[str, Any] = {
                "temperature": 0,
                "top_p": 1,
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json",
            }
            if use_schema is not None:
                cfg_dict["response_schema"] = use_schema
            cfg_to_use = cfg_dict

        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt_to_use,
                config=cfg_to_use,
            )

            out_text, finish_reason, safety_info = _extract_response_text(resp)
            last_raw = out_text
            last_finish_reason = finish_reason

            if safety_info:
                _log_event(
                    "llm_safety_blocked",
                    {
                        "model": model,
                        "attempt": attempt,
                        "safety_info": safety_info,
                    },
                )
                raise RuntimeError(f"Response blocked by safety filters: {safety_info}")

            if finish_reason and "MAX_TOKENS" in finish_reason.upper():
                _log_event(
                    "llm_max_tokens_hit",
                    {
                        "model": model,
                        "attempt": attempt,
                        "finish_reason": finish_reason,
                        "max_tokens": max_tokens,
                        "raw_len": len(out_text),
                    },
                )

            parsed = getattr(resp, "parsed", None)
            if parsed is not None:
                if isinstance(parsed, dict):
                    return parsed
                if hasattr(parsed, "model_dump"):
                    return parsed.model_dump()
                if isinstance(parsed, list):
                    return {"value": parsed}
                return json.loads(json.dumps(parsed, ensure_ascii=False))

            if not out_text or not out_text.strip():
                raise ValueError("Empty model output")

            return _extract_json(out_text)

        except Exception as e:
            last_err = e
            _log_event(
                "llm_json_parse_failed",
                {
                    "model": model,
                    "attempt": attempt,
                    "err": str(e),
                    "finish_reason": last_finish_reason,
                    "raw_len": len(last_raw) if last_raw else 0,
                    "raw_snippet": _truncate(last_raw, 900),
                    "schema_used": use_schema is not None,
                    "will_retry_without_schema": attempt == 0 and retries > 0,
                },
            )
            continue

    raise RuntimeError(
        f"LLM JSON output failed after {retries + 1} attempts: {last_err}. raw={_truncate(last_raw, 400)}"
    )


def _call_llm_with_fallback(
    model: str,
    prompt: str,
    schema: Optional[Dict[str, Any]],
    validator: callable,
    retries: int,
    stage_name: str,
    event_id: str,
    order_id: str,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> Dict[str, Any]:
    try:
        result = _gen_json(model, prompt, max_tokens=max_tokens, response_schema=schema, retries=retries)
        validator(result)
        return result
    except Exception as e_schema:
        _log_event(
            f"{stage_name}_schema_or_parse_failed_fallback",
            {"event_id": event_id, "order_id": order_id, "err": str(e_schema)},
        )
        result = _gen_json(model, prompt, max_tokens=max_tokens, response_schema=None, retries=retries)
        validator(result)
        return result


# -----------------------------
# Prompts
# -----------------------------
STAGE1_BUCKETS = [
    "ETHNICITY_RELATED",
    "POLITICS_RELATED",
    "GENDER_OPPOSITION_AND_RELATED",
    "UNRELATED_TO_GAME",
    "STORE_STAR_SITUATION",
    "HIGHLY_SENSITIVE_OTHER",
    "NONE",
    "UNCERTAIN",
]


def _stage1_prompt(payload: Dict[str, Any], title: str, body: str) -> str:
    language = payload.get("language") or ""
    territory = payload.get("territory") or ""
    game_biz = payload.get("game_biz") or ""
    rating = payload.get("rating")

    return f"""
You are a strict classifier for sensitive topics in app store reviews.
Output must be valid JSON and match schema exactly. No extra keys.

Classify the review into EXACTLY ONE bucket:

**ETHNICITY_RELATED**: Mentions race/ethnicity/nationality stereotypes, racism, discrimination, ethnic slurs, "anti-[group]", "x people are…"
Examples:
- "This game is racist against [group], disgusting."
- "They portray [nationality] as evil, boycott them."

**POLITICS_RELATED**: Real-world politics, governments, territorial disputes, propaganda, sanctions, wars, political ideologies
Examples:
- "Taiwan/HK is a country, fix your map."
- "Stop supporting [government], uninstalling."

**GENDER_OPPOSITION_AND_RELATED**: Sexism, feminism/anti-feminism, gender identity fights, "woke", misogyny, "men vs women"
Examples:
- "Too woke, pushing feminist agenda."
- "Women are portrayed as objects, gross."
NOT gender opposition (classify as NONE instead):
- Gaming slang: "waifu bait", "husbando collector" - these are game design criticisms, not gender politics
- Character design preferences without actual gender/political arguments

**UNRELATED_TO_GAME**: Complaints totally unrelated to the game itself (delivery issues, personal life, random spam, political rants not tied to game)
IMPORTANT: Device compatibility, technical issues, crashes, and lag ARE related to the game - classify as NONE, not UNRELATED.
Examples of UNRELATED:
- "My parcel didn't arrive."
- "This company is a scam." (generic company complaint)
Examples that ARE related (use NONE, not UNRELATED):
- "Game doesn't work on my device"
- "Game crashes/lags"

**STORE_STAR_SITUATION**: Disputes about store rating/review moderation, star changes, review removed, can't edit review, store UI problems, claims that reviews/stars are "fake"
Examples:
- "Google deleted my 1-star review."
- "My rating keeps changing / review not showing."
- "All these 5-star reviews are fake bots."

**HIGHLY_SENSITIVE_OTHER**: Any of the following sensitive categories:
[History & Real-World Politics]
- Boundary/territorial disputes, negative/stereotypical portrayals of national images
- Glorification, trivialization, or mockery of war conflicts and historical suffering
- Fictional settings reflecting real-world political issues, political stance/ideological inclination

[Inclusivity & Diversity]
- Controversies surrounding diversity and political correctness
- Issues with gender, race proportions, sexual orientation representation

[Culture & Religion]
- Cultural stereotypes (simplified/distorted depictions of cultures)
- Cultural appropriation, religious offense, sacred symbols misuse
- Associating religious content with violence, sexuality, or other sensitive themes

[Sexuality & Violence]
- Sexual objectification, characters in revealing clothing, gender stereotypes
- Explicit/implicit portrayals of sexual content, sexual violence
- Violence from real events, simulation of violent crimes, extremism

[Other Specific Issues]
- Game content censorship debates
- Privacy and data handling concerns
- Labor disputes, strikes, union issues, voice actor strikes
  → ANY mention of "strike" or "unvoiced due to strike" triggers this category
- Character diversity complaints ("lack of diversity", "no diversity", "character diversity")
  → ANY mention triggers this, regardless of tone

**NONE**: None of the above sensitive categories apply - safe to proceed to Stage 2

**UNCERTAIN**: Ambiguous; could be sensitive but not sure - route to human review

Decision rules:
- If ANY sensitive/political/identity/religion/war/privacy/censorship/extremism content appears → pick the matching sensitive bucket
- If unclear or ambiguous → UNCERTAIN (never guess NONE when uncertain)
- If clearly none of the sensitive categories apply → NONE

Review:
game_biz: {game_biz}
rating: {rating}
territory: {territory}
language: {language}
title: {title}
body: {body}

Return JSON:
{{
  "bucket": "NONE",
  "confidence": 0.0,
  "rationale": "short reason, <= 20 words"
}}
""".strip()


def _stage2_prompt(payload: Dict[str, Any], game: str, allowed_topics: List[str], title: str, body: str) -> str:
    language = payload.get("language") or ""
    rating = payload.get("rating")

    topics_str = ", ".join([f'"{t}"' for t in allowed_topics])

    return f"""
You are a strict classifier for Google Play review issues.
Return valid JSON only. No extra keys.

=== CRITICAL RULES ===
1. ALL topics use SPECIFIC_ISSUE, EXCEPT these three which use GENERAL_ISSUE:
   - "恶意差评 Malicious Low Score" → GENERAL_ISSUE
   - "具体问题的差评 Reasonable Low Score with specific problems" → GENERAL_ISSUE
   - "无缘由差评 Unreasonable Low Score" → GENERAL_ISSUE

2. 恶意差评 is the DEFAULT for game design complaints. Use 具体问题的差评 ONLY for pay-to-win / monetization design complaints OR network/connection issues without an explicit error code (see Step 2b).

3. MULTILINGUAL: Reviews may be in ANY language. The keywords listed below are English examples only.

   Always match on the MEANING/CONCEPT, not the exact English word. For example, Russian "место" = "space",
   Chinese "内存" = "memory", Spanish "descarga" = "download", etc. Apply this to ALL steps below.

4. ROOT CAUSE RULE: When a review mentions MULTIPLE problems in a chain (e.g., "game crashed → couldn't log in → missed event rewards"), classify based on the ROOT CAUSE of all the issues, NOT the downstream effects. Trace the causal chain back to the origin problem and pick the template for that root cause. Examples: if a crash caused a login failure which caused a missed event, the root cause is the crash — use Device Issues (if device context) or Account issues (if no device context). If network/connection caused a login failure, the root cause is network — use 具体问题的差评, NOT Device Issues or Account issues.

=== CLASSIFICATION LOGIC (apply STRICTLY in order) ===

STEP 1: SPECIFIC_ISSUE topics (check these first)

1a. ERROR → "Error Code"
    If review explicitly mentions "error" or "error code" AND it is about the GAME itself (in-game error, launch error, game error code)
    → Error Code
    Do NOT use Error Code for:
    - Complaints about ads, YouTube, or external apps (e.g., "your ad breaks my youtube") → use 恶意差评
    - General negativity that doesn't mention an actual error → use 恶意差评
    - Vague use of "error" not referring to a game error code/message → use the appropriate topic
    Examples: "Login error", "error code 1001", "game shows error when I open it"

1b. DOWNLOAD → "Download Issue"
    If review contains "download" or "install" → Download Issue
    EXCEPT: if the download/install problem is CAUSED BY the game being too large (mentions size, storage, GB, "too big", "too heavy", "takes too much space"), apply the ROOT CAUSE RULE — the root cause is Big Size, not the download itself. Use Big Size instead.

1c. STORAGE → "Big Size"
    If review mentions storage/space ("storage", "space", "memory", "too big", "gb", "size", "too heavy", "too large")
    → Big Size
    This includes cases where download is mentioned but the ROOT CAUSE is the game's large size (e.g., "can't download, too big", "download takes forever because of the size", "game eats all my storage").
    IMPORTANT OVERRIDE: If the review mentions inability to play/load/run the game ("can't play", "can't load", "won't run", "unable to play", "can't even play"), Device Issues ALWAYS takes precedence over Big Size — even if storage/size keywords are present. The user needs the Device Issues template (system requirements advice), not storage tips.

1d. DEVICE ISSUES → "Device Issues" (STRICT CRITERIA)
    If storage keywords are present but the user says they can't play/load/run the game → Device Issues takes precedence (storage is just the symptom; the user needs system requirements advice)
    REQUIRES explicit device context. Use ONLY if review has:
    - Device words: "phone", "tablet", "device", "my [device name]", "older devices"
    - Optimization: "optimize", "optimisation", "optimization"
    - Requirements: "requirements", "specs", "system requirements"
    - GPU terms: "shaders", "compiling shaders"
    - "can't play" / "couldn't play" / "unable to play" as core complaint — even in longer reviews, if the user's main point is they CANNOT play the game (not just disliking it), use Device Issues regardless of review length or whether a device is explicitly named
    - Game crashes as ROOT CAUSE: if the review describes the game crashing/freezing and that crash leads to other issues (can't log in, missed events, etc.), the root cause is the crash — use Device Issues.
    NOTE: If a review mentions a device but the root cause is clearly network/connection (not device performance), use 具体问题的差评 instead. Example: "network error on my tablet" → root cause is network, device mention is incidental → 具体问题的差评. But "game lags on my phone" → root cause is device performance → Device Issues.

    DO NOT use Device Issues for:
    - "SLOW" when referring to dialogue/gameplay pacing → use 恶意差评
    - "Technical issues" without device context AND without crash root cause → use Account issues
    - "Plays terribly" (subjective quality criticism) → use 恶意差评
    - Network/connection issues without device context → use 具体问题的差评

1e. ACCOUNT/TECH ISSUES → "Account issues"
    Use for:
    - Login/account problems that are NOT caused by network/connection/crash: "password", "OTP", "verification", "email code", "guest account", "account banned", "account lost"
    - Vague tech WITHOUT device context and WITHOUT network/crash root cause: "technical issues", "can't access"
    IMPORTANT: If login failure is CAUSED BY network/connection issues, use 具体问题的差评 instead (the login problem is a downstream symptom; the root cause is network). If login failure is CAUSED BY game crashes, use Device Issues instead (the root cause is the crash).

1f. PAYMENT → "Payment issues"
    "payment", "purchase", "refund", "charge", "money"

1g. GACHA/DROP → "Gacha/drop"
    Use when the review is primarily about gacha pull outcomes — especially spending money/resources but NOT getting the character or item they want:
    - Gacha mechanics: pity, banners, wishing/warping/signal search, pulling, drop rates
    - Specific pull/drop outcomes: "spent X but didn't get Y", "didn't get X", "X не выпал", "lost 50/50"
    - Spending money on gacha: "spent $100 and got nothing", "wasted money on pulls", "saved for months and didn't get the character"
    - Rewards complaints: "bad rewards", "malas recompensas", "stingy rewards", "reduced rewards"
    - Economy/currency issues: "not enough primogems/stellar jade/polychrome/coins/gems", "prices too high"
    This includes both factual reports AND complaints/rants about these topics — even with negative tone.
    Do NOT use for "pay to win" / monetization DESIGN complaints (e.g., "this game is pay to win", "only whales can compete") → use 具体问题的差评 instead
    Do NOT use for reviews where gacha/rewards is only a brief mention inside a broader game design rant → use 恶意差评

1h. SUGGESTIONS → "Suggestions" (SPECIFIC_ISSUE)
    Use if the review contains EITHER:
    (a) Polite request keywords: "wish", "would be nice", "please add", "I hope", "suggestion"
    (b) Direct action requests that ask to ADD, REMOVE, or CHANGE a specific game feature:
        - Imperative verbs: "add X", "remove X", "delete X", "change X", "bring back X", "make X", "let us X", "allow X"
        - These apply in ANY language (e.g. Russian "удалите" = "remove", "добавьте" = "add", Chinese "加" = "add", "删" = "remove/delete")
        - The review must be requesting a specific feature change, not just complaining

    Do NOT use Suggestions for:
    - Complaints without a specific request: "too much X", "not enough X", "X is bad" → use 恶意差评
    - Hostile/vague negativity without actionable request: "trash game", "boring" → use 恶意差评

    Examples:
    - "I wish there was a skip button" → Suggestions (polite keyword "wish")
    - "Remove the resin/trailblaze power/battery system" → Suggestions (direct action request: "remove X")
    - "удалите астральный предел" → Suggestions (imperative "удалите" = "remove X", HSR stamina system)
    - "Add a pity counter display" → Suggestions (direct action request: "add X")
    - "Still no skip button" → 恶意差评 (complaint, no request or action verb)
    - "Stingy gacha and extremely grindy" → Gacha/drop (primarily about gacha/rewards)

STEP 2: GENERAL_ISSUE topics (only if no SPECIFIC_ISSUE matches)

2a. GAME DESIGN COMPLAINTS → "恶意差评 Malicious Low Score" (GENERAL_ISSUE)
    DEFAULT for all game design feedback that isn't explicitly polite:
    - Any complaints about combat, story, quests, pacing, general gameplay
    - Detailed feedback with negative/complaining tone
    - Hostile or discouraging language
    - Pure hostility: "trash", "garbage", "worst game ever"
    - Discouraging others: "don't play", "waste of time"
    - Short reviews that clearly express a negative opinion about the game
      (e.g. "bad game", "juego malo", "垃圾游戏", "jogo ruim")
    - Dismissive single words that express contempt: "meh", "bad", "mid", "trash", "nah"

    Use 恶意差评 unless the review uses polite language like "wish", "please", "I hope".
    Do NOT use 恶意差评 if the review text is actually POSITIVE (e.g. "Very impressive", "Great game") with a low star rating → use 无缘由差评 instead.

2b. PAY-TO-WIN / MONETIZATION DESIGN / NETWORK ISSUES → "具体问题的差评 Reasonable Low Score with specific problems" (GENERAL_ISSUE)
    Use for:
    (a) Monetization DESIGN complaints: "pay to win", "p2w", "money grab", "cash grab",
        "aggressive monetization", "only whales can compete", "need to spend money to progress",
        "free players can't compete".
        These are complaints about the game's monetization DESIGN, not about specific gacha pull outcomes.
        If the user is upset about spending money and NOT getting a specific character/item → use Gacha/drop instead.
    (b) Network/connection issues WITHOUT an explicit error code: "connection lost",
        "network error", "can't connect", "server timeout", "timed out", "lag",
        "high ping", "connection error", "disconnect", "connect lost". These are server/infrastructure
        problems, not device problems.
        IMPORTANT: If the review mentions a specific error code alongside network issues
        (e.g. "network error code 4206"), use Error Code instead (already matched in Step 1a).

2c. SHORT/VAGUE/CONTRADICTORY → "无缘由差评 Unreasonable Low Score" (GENERAL_ISSUE)
    Use for reviews where the low rating doesn't match the text:
    - Truly meaningless: random characters, single non-descriptive words ("ok", "...")
    - Mixed positive/negative with no clear direction
    - POSITIVE text with low star rating: e.g. "Very impressive", "Great game", "Love it" with 1-2 stars — the text contradicts the rating, so the low score is unreasonable
    NOTE: If a short review expresses a clear negative opinion about the game
    (e.g. "bad game", "juego malo", "garbage"), use 恶意差评 instead.

=== EXAMPLES ===
| Review | Topic | Why |
|--------|-------|-----|
| "Login error" | Error Code | game-related error |
| "ad breaks other apps" | 恶意差评 | not a game error |
| "can't play" | Device Issues | short, could be compatibility |
| "crashes on title screen" | Account issues | crash WITHOUT device |
| "installed but freezes at first cinematic, can't play at all" | Device Issues | completely unplayable from start |
| "crashes on my phone" | Device Issues | crash WITH device |
| "Technical issues preventing access" | Account issues | vague, no device |
| "game restarts at 99%; compiling shaders" | Device Issues | "shaders" = GPU term |
| "Lagging hard need optimisation" | Device Issues | asks for optimization |
| "I wish there was a skip button" | Suggestions | polite keyword ("wish") |
| "Please add more rewards" | Suggestions | polite keyword ("please") |
| "Remove the resin/trailblaze power/battery system" | Suggestions | direct action request ("remove X") |
| "удалите астральный предел" | Suggestions | imperative request ("удалите" = "remove", HSR) |
| "SLOW dialogue, unskippable cutscenes" | 恶意差评 | complaint, no request or action verb |
| "Still no skip button on long quests" | 恶意差评 | complaint, no action verb |
| "Stingy gacha and extremely grindy" | Gacha/drop | primarily about gacha/rewards |
| "Spent $200 and didn't get the character" | Gacha/drop | spent money, didn't get desired char |
| "This game is pay to win" | 具体问题的差评 | monetization design complaint, not gacha outcome |
| "Only whales can compete, p2w trash" | 具体问题的差评 | pay-to-win design complaint |
| "Game crashed, couldn't log in, missed event" | Device Issues | root cause is crash (trace causal chain) |
| "can't log in, always says connect lost timed out" | 具体问题的差评 | root cause is network/connection, not device |
| "server keeps disconnecting me" | 具体问题的差评 | network/server issue, no error code |
| "network error code 4206 when logging in" | Error Code | explicit error code mentioned |
| "high ping and lag every day" | 具体问题的差评 | network/connectivity issue |
| "Too many quests; feels like a chore" | 恶意差评 | complaining tone |
| "gg whales" | 恶意差评 | hostile, no detail |
| "trash game don't play" | 恶意差评 | hostile, discouraging, no detail |
| "bad game" / "juego malo" | 恶意差评 | short but clear negative opinion |
| "bad" | 恶意差评 | dismissive, expresses negative opinion |
| "Very impressive" (1-star) | 无缘由差评 | positive text contradicts low rating |
| "Game freezes and kicks out; black screen" | Account issues | tech issues, no device |
| "can't download, game is too big" | Big Size | download caused by game size → root cause is Big Size |
| "download it...eat my gb space" | Big Size | download caused by large size → root cause is Big Size |
| "download stuck at 50%" | Download Issue | download problem, no size/storage cause |
| "can't play it on my phone, takes all my storage" | Device Issues | can't play + device = Device Issues, even with storage mention |

Available topics: [{topics_str}]

Review:
game: {game}
rating: {rating}
language: {language}
title: {title}
body: {body}

ADDITIONAL OUTPUT (optional but preferred):
- key_phrases: Extract 2-5 specific phrases FROM THE REVIEW TEXT that most strongly influenced your classification. Use the original language.
- aspects: Break the review into 1-3 aspects (e.g. stability, performance, UX, billing, content) with sentiment per aspect and the evidence quote.
- confidence_factors: Report whether the review has mixed_signals (both positive and negative), language_clarity (clear/ambiguous/sarcastic), and text_length (sufficient/short/very_short).

Return JSON:
{{
  "issue_type": "GENERAL_ISSUE or SPECIFIC_ISSUE",
  "topic": "exactly one from allowed list",
  "confidence": 0.0-1.0,
  "rationale": "short reason, <= 20 words",
  "key_phrases": ["phrase1", "phrase2"],
  "aspects": [{{"aspect": "stability", "sentiment": "negative", "evidence": "quote"}}],
  "confidence_factors": {{"mixed_signals": false, "language_clarity": "clear", "text_length": "sufficient"}}
}}
""".strip()


def _stage1_response_schema() -> Dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "bucket": {"type": "STRING", "enum": STAGE1_BUCKETS},
            "confidence": {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "STRING"},
        },
        "required": ["bucket", "confidence", "rationale"],
        "propertyOrdering": ["bucket", "confidence", "rationale"],
    }


def _stage2_response_schema(allowed_topics: List[str]) -> Dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "issue_type": {"type": "STRING", "enum": ["SPECIFIC_ISSUE", "GENERAL_ISSUE"]},
            "topic": {"type": "STRING", "enum": allowed_topics},
            "confidence": {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "STRING"},
            "key_phrases": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "2-5 specific phrases from the review that drove classification",
            },
            "aspects": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "aspect": {"type": "STRING"},
                        "sentiment": {"type": "STRING", "enum": ["positive", "negative", "neutral"]},
                        "evidence": {"type": "STRING"},
                    },
                    "required": ["aspect", "sentiment", "evidence"],
                },
                "description": "Aspect-level sentiment breakdown",
            },
            "confidence_factors": {
                "type": "OBJECT",
                "properties": {
                    "mixed_signals": {"type": "BOOLEAN"},
                    "language_clarity": {"type": "STRING", "enum": ["clear", "ambiguous", "sarcastic"]},
                    "text_length": {"type": "STRING", "enum": ["sufficient", "short", "very_short"]},
                },
                "required": ["mixed_signals", "language_clarity", "text_length"],
            },
        },
        "required": ["issue_type", "topic", "confidence", "rationale"],
        "propertyOrdering": ["issue_type", "topic", "confidence", "rationale",
                             "key_phrases", "aspects", "confidence_factors"],
    }


# -----------------------------
# Template and validation helpers
# -----------------------------
def _lang_to_template_key(lang_code: str) -> Tuple[str, bool]:
    if not lang_code:
        return DEFAULT_TEMPLATE_LANG, True
    k = LANG_MAP.get(lang_code.strip().lower())
    return (k, False) if k else (DEFAULT_TEMPLATE_LANG, True)


def _select_template(game: str, issue_type: str, topic: str, lang_key: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    exact_key = (game, issue_type, topic, lang_key)
    if exact_key in TEMPLATE_INDEX:
        return TEMPLATE_INDEX[exact_key], False

    fallback_key = (game, issue_type, topic, DEFAULT_TEMPLATE_LANG)
    if fallback_key in TEMPLATE_INDEX:
        return TEMPLATE_INDEX[fallback_key], True

    return None, False


def _validate_confidence_range(conf: Any, stage_name: str) -> None:
    """Validate confidence is between 0.0 and 1.0."""
    try:
        c = float(conf)
        if c < 0.0 or c > 1.0:
            raise ValueError(f"{stage_name}_confidence_out_of_range")
    except (TypeError, ValueError) as e:
        if "out_of_range" not in str(e):
            raise ValueError(f"{stage_name}_invalid_confidence")
        raise


def _validate_stage1(stage1: Dict[str, Any]) -> None:
    if "bucket" not in stage1:
        raise ValueError("stage1_missing_bucket")
    if stage1.get("bucket") not in STAGE1_BUCKETS:
        raise ValueError(f"stage1_invalid_bucket:{stage1.get('bucket')}")
    try:
        _validate_confidence_range(stage1.get("confidence"), "stage1")
    except Exception as e:
        _log_event("stage1_confidence_validation_warning", {"err": str(e), "confidence": stage1.get("confidence")})


def _validate_stage2(stage2: Dict[str, Any], allowed_topics: List[str], game: str = "") -> None:
    it = (stage2.get("issue_type") or "").strip()
    tp = (stage2.get("topic") or "").strip()

    if it not in ("SPECIFIC_ISSUE", "GENERAL_ISSUE"):
        raise ValueError(f"stage2_invalid_issue_type:{it}")
    if tp not in allowed_topics:
        raise ValueError(f"stage2_invalid_topic:{tp}")

    # Auto-correct issue_type when it doesn't match the template-defined mapping
    if game and tp:
        correct_it = TOPIC_ISSUE_TYPE.get((game, tp))
        if correct_it and it != correct_it:
            _log_event("stage2_issue_type_autocorrected", {
                "game": game, "topic": tp,
                "ai_issue_type": it, "corrected_issue_type": correct_it,
            })
            stage2["issue_type"] = correct_it

    _validate_confidence_range(stage2.get("confidence"), "stage2")


def _max_reply_chars() -> int:
    try:
        return int(
            TEMPLATE_REGISTRY.get("meta", {})
            .get("template_constraints", {})
            .get("google_play_max_chars", 350)
        )
    except Exception:
        return 350


# -----------------------------
# Firestore REST writer
# -----------------------------
_HTTP = requests.Session()
_http_retry = Retry(total=2, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
_HTTP.mount("https://", HTTPAdapter(max_retries=_http_retry, pool_maxsize=10))
_HTTP.mount("http://", HTTPAdapter(max_retries=_http_retry, pool_maxsize=10))
_FS_CREDS = None
_FS_PROJECT = None
_FS_AUTH_REQ = GoogleAuthRequest()
_FS_LOCK = threading.Lock()


def _fs_init_auth() -> Tuple[Any, str]:
    global _FS_CREDS, _FS_PROJECT
    if _FS_CREDS is not None and _FS_PROJECT:
        return _FS_CREDS, _FS_PROJECT

    creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/datastore"])
    if not project:
        project = _project_id()
    if not project:
        raise RuntimeError("Unable to determine project id for Firestore (ADC project is empty).")

    _FS_CREDS = creds
    _FS_PROJECT = project
    return _FS_CREDS, _FS_PROJECT


def _fs_get_token() -> str:
    creds, _ = _fs_init_auth()
    if not creds.valid or creds.expired:
        with _FS_LOCK:
            if not creds.valid or creds.expired:
                creds.refresh(_FS_AUTH_REQ)
    if not creds.token:
        raise RuntimeError("Failed to obtain access token for Firestore REST.")
    return creds.token


def _fs_safe_doc_id(doc_id: str) -> str:
    doc_id = (doc_id or "").strip() or f"evt_{int(time.time())}"
    doc_id = doc_id.replace("/", "_")
    doc_id = re.sub(r"[^A-Za-z0-9._-]", "_", doc_id)
    return doc_id[:900]


def _fs_value(v: Any) -> Dict[str, Any]:
    if v is None:
        return {"nullValue": None}
    if isinstance(v, bool):
        return {"booleanValue": v}
    if isinstance(v, int):
        return {"integerValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": float(v)}
    if isinstance(v, str):
        return {"stringValue": v}
    if isinstance(v, dict):
        return {"mapValue": {"fields": {str(k): _fs_value(vv) for k, vv in v.items()}}}
    if isinstance(v, list):
        return {"arrayValue": {"values": [_fs_value(x) for x in v]}}
    return {"stringValue": str(v)}


def _fs_parse_value(v: Dict[str, Any]) -> Any:
    """Inverse of _fs_value(): convert Firestore REST value back to Python."""
    if "nullValue" in v:
        return None
    if "booleanValue" in v:
        return v["booleanValue"]
    if "integerValue" in v:
        return int(v["integerValue"])
    if "doubleValue" in v:
        return v["doubleValue"]
    if "stringValue" in v:
        return v["stringValue"]
    if "mapValue" in v:
        fields = v["mapValue"].get("fields", {})
        return {k: _fs_parse_value(vv) for k, vv in fields.items()}
    if "arrayValue" in v:
        values = v["arrayValue"].get("values", [])
        return [_fs_parse_value(x) for x in values]
    return None


def _fs_doc_url(project: str, database: str, collection: str, doc_id: str) -> str:
    base = f"https://firestore.googleapis.com/v1/projects/{project}/databases/{database}/documents"
    return f"{base}/{collection}/{doc_id}"


def _firestore_read_existing_result(event_id: str) -> Optional[Dict[str, Any]]:
    """Read existing processing result from Firestore. Returns result dict or None."""
    if not FIRESTORE_ENABLED:
        return None
    try:
        _, project = _fs_init_auth()
        token = _fs_get_token()
        doc_id = _fs_safe_doc_id(event_id)
        url = _fs_doc_url(project, FIRESTORE_DATABASE, FIRESTORE_COLLECTION, doc_id)
        headers = {"Authorization": f"Bearer {token}"}
        resp = _HTTP.get(url, headers=headers, timeout=FIRESTORE_WRITE_TIMEOUT_SEC)
        if resp.status_code != 200:
            return None
        doc = resp.json()
        fields = doc.get("fields", {})
        parsed = {k: _fs_parse_value(v) for k, v in fields.items()}
        if not parsed.get("action"):
            return None
        return parsed
    except Exception as e:
        _log_event("firestore_read_failed", {"event_id": event_id, "err": str(e)})
        return None


def _firestore_write_review_event_best_effort(
    *,
    event_id: str,
    payload: Dict[str, Any],
    result: Dict[str, Any],
    latency_ms: int,
) -> None:
    if not FIRESTORE_ENABLED:
        return

    try:
        _, project = _fs_init_auth()
        token = _fs_get_token()

        doc_id = _fs_safe_doc_id(event_id)

        stage1 = result.get("stage1") or {}
        stage2 = result.get("stage2") or {}
        template = result.get("template") or {}

        record: Dict[str, Any] = {
            "event_id": event_id,
            "order_id": payload.get("order_id"),
            "event_type": payload.get("event_type"),
            "store_type": payload.get("store_type"),
            "game_biz": payload.get("game_biz"),
            "app_id": payload.get("app_id"),
            "review_id": payload.get("review_id"),
            "rating": payload.get("rating"),
            "language": payload.get("language"),
            "territory": payload.get("territory"),
            "review_at": payload.get("review_at"),
            "review_title": payload.get("title"),
            "review_body": payload.get("body"),
            "action": result.get("action"),
            "gate_result": result.get("gate_result"),
            "gate_reason": result.get("gate_reason"),
            "game": result.get("game"),
            "error": result.get("error") or result.get("reason"),
            "template_id": template.get("template_id"),
            "reply_text": template.get("reply_text"),
            "stage1_bucket": stage1.get("bucket"),
            "stage1_confidence": stage1.get("confidence"),
            "stage1_rationale": stage1.get("rationale"),
            "stage2_issue_type": stage2.get("issue_type"),
            "stage2_topic": stage2.get("topic"),
            "stage2_confidence": stage2.get("confidence"),
            "stage2_rationale": stage2.get("rationale"),
            "stage2_key_phrases": stage2.get("key_phrases") if stage2.get("key_phrases") else None,
            "stage2_aspects": stage2.get("aspects") if stage2.get("aspects") else None,
            "stage2_confidence_factors": stage2.get("confidence_factors") if stage2.get("confidence_factors") else None,
            "latency_ms": latency_ms,
            "ingested_at": _utc_iso_now(),
        }

        body = {"fields": {k: _fs_value(v) for k, v in record.items()}}

        url = _fs_doc_url(project, FIRESTORE_DATABASE, FIRESTORE_COLLECTION, doc_id)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }

        resp = _HTTP.patch(url, headers=headers, data=json.dumps(body), timeout=FIRESTORE_WRITE_TIMEOUT_SEC)

        if 200 <= resp.status_code < 300:
            _log_event(
                "firestore_write_ok",
                {
                    "event_id": event_id,
                    "database": FIRESTORE_DATABASE,
                    "collection": FIRESTORE_COLLECTION,
                    "doc_id": doc_id,
                },
            )
            return

        _log_event(
            "firestore_write_failed",
            {
                "event_id": event_id,
                "database": FIRESTORE_DATABASE,
                "collection": FIRESTORE_COLLECTION,
                "doc_id": doc_id,
                "http_status": resp.status_code,
                "resp_snippet": _truncate(resp.text, 900),
            },
        )

    except Exception as e:
        _log_event(
            "firestore_write_failed",
            {
                "event_id": event_id,
                "database": FIRESTORE_DATABASE,
                "collection": FIRESTORE_COLLECTION,
                "err": str(e),
            },
        )


# -----------------------------
# Topic analysis (sub-issue drill-down)
# -----------------------------
TOPIC_ANALYSIS_COLLECTION = "topic_analysis"
TOPIC_ANALYSIS_CACHE_TTL_SEC = 24 * 60 * 60  # 24 hours
TOPIC_ANALYSIS_MAX_REVIEWS = 100


def _topic_cache_key(topic: str, game: Optional[str], language: Optional[str]) -> str:
    raw = f"{topic}|{game or ''}|{language or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _fs_read_topic_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Read cached topic analysis from Firestore. Returns dict or None."""
    if not FIRESTORE_ENABLED:
        return None
    try:
        _, project = _fs_init_auth()
        token = _fs_get_token()
        url = _fs_doc_url(project, FIRESTORE_DATABASE, TOPIC_ANALYSIS_COLLECTION, cache_key)
        headers = {"Authorization": f"Bearer {token}"}
        resp = _HTTP.get(url, headers=headers, timeout=FIRESTORE_WRITE_TIMEOUT_SEC)
        if resp.status_code != 200:
            return None
        doc = resp.json()
        fields = doc.get("fields", {})
        parsed = {k: _fs_parse_value(v) for k, v in fields.items()}
        # Check TTL
        expires_at = parsed.get("ttl_expires_at", "")
        if expires_at and expires_at > _utc_iso_now():
            return parsed
        return None
    except Exception as e:
        _log_event("topic_cache_read_failed", {"cache_key": cache_key, "err": str(e)})
        return None


def _fs_write_topic_cache(cache_key: str, data: Dict[str, Any]) -> None:
    """Write topic analysis result to Firestore cache."""
    if not FIRESTORE_ENABLED:
        return
    try:
        _, project = _fs_init_auth()
        token = _fs_get_token()
        url = _fs_doc_url(project, FIRESTORE_DATABASE, TOPIC_ANALYSIS_COLLECTION, cache_key)
        body = {"fields": {k: _fs_value(v) for k, v in data.items()}}
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        resp = _HTTP.patch(url, headers=headers, data=json.dumps(body), timeout=FIRESTORE_WRITE_TIMEOUT_SEC)
        if 200 <= resp.status_code < 300:
            _log_event("topic_cache_write_ok", {"cache_key": cache_key})
        else:
            _log_event("topic_cache_write_failed", {"cache_key": cache_key, "http_status": resp.status_code})
    except Exception as e:
        _log_event("topic_cache_write_failed", {"cache_key": cache_key, "err": str(e)})


def _fs_query_reviews_by_topic(
    topic: str,
    game: Optional[str] = None,
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query review_events by stage2_topic using Firestore REST :runQuery."""
    _, project = _fs_init_auth()
    token = _fs_get_token()

    base = f"https://firestore.googleapis.com/v1/projects/{project}/databases/{FIRESTORE_DATABASE}/documents"
    url = f"{base}:runQuery"

    # Build structured query with field filters
    field_filters = [
        {
            "fieldFilter": {
                "field": {"fieldPath": "stage2_topic"},
                "op": "EQUAL",
                "value": {"stringValue": topic},
            }
        },
        # Only include real CSC webhook reviews (event_id prefix "evt_gp_")
        # '_' is ASCII 95, '`' is ASCII 96 — so "evt_gp`" is the first string
        # that doesn't match the "evt_gp_" prefix (lexicographic range filter).
        {
            "fieldFilter": {
                "field": {"fieldPath": "event_id"},
                "op": "GREATER_THAN_OR_EQUAL",
                "value": {"stringValue": "evt_gp_"},
            }
        },
        {
            "fieldFilter": {
                "field": {"fieldPath": "event_id"},
                "op": "LESS_THAN",
                "value": {"stringValue": "evt_gp`"},
            }
        },
    ]
    if game:
        field_filters.append({
            "fieldFilter": {
                "field": {"fieldPath": "game"},
                "op": "EQUAL",
                "value": {"stringValue": game},
            }
        })
    if language:
        field_filters.append({
            "fieldFilter": {
                "field": {"fieldPath": "language"},
                "op": "EQUAL",
                "value": {"stringValue": language},
            }
        })

    if len(field_filters) == 1:
        where_clause = field_filters[0]
    else:
        where_clause = {"compositeFilter": {"op": "AND", "filters": field_filters}}

    query_body = {
        "structuredQuery": {
            "from": [{"collectionId": FIRESTORE_COLLECTION}],
            "where": where_clause,
            "limit": TOPIC_ANALYSIS_MAX_REVIEWS,
        }
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    resp = _HTTP.post(url, headers=headers, data=json.dumps(query_body), timeout=15)
    resp.raise_for_status()

    results = []
    for item in resp.json():
        doc = item.get("document")
        if not doc:
            continue
        fields = doc.get("fields", {})
        parsed = {k: _fs_parse_value(v) for k, v in fields.items()}
        review_body = parsed.get("review_body") or ""
        if not review_body.strip():
            continue
        results.append({
            "review_body": review_body,
            "language": parsed.get("language") or "unknown",
            "rating": parsed.get("rating"),
            "game": parsed.get("game") or "unknown",
        })

    return results


def _build_topic_analysis_prompt(topic: str, reviews: List[Dict[str, Any]]) -> str:
    review_lines = []
    for i, r in enumerate(reviews, 1):
        lang = r.get("language", "unknown")
        rating = r.get("rating", "?")
        body = _truncate(r["review_body"], 500)
        review_lines.append(f"[{i}] lang={lang} rating={rating}: {body}")

    reviews_block = "\n".join(review_lines)
    n = len(reviews)

    return f"""You are analyzing Google Play reviews all classified under the topic "{topic}".
Identify the specific sub-issues within this topic.

RULES:
1. Reviews are MULTILINGUAL. Analyze ALL languages, group by MEANING not language.
2. Identify 3-7 distinct sub-issues — specific, actionable complaints.
3. Every review must belong to exactly one sub-issue. Counts must sum to {n}.
4. Pick 2-3 representative quotes per sub-issue (keep original language, max 120 chars each).
5. Name sub-issues in English, clearly and specifically (e.g., "Crashes during co-op gameplay" not "Crash issues").
6. Order by count descending.
7. Write a 1-2 sentence summary of the overall pattern.

REVIEWS ({n} total, topic: "{topic}"):
{reviews_block}

Return JSON with the schema provided."""


TOPIC_ANALYSIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "sub_issues": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "count": {"type": "INTEGER"},
                    "percentage": {"type": "NUMBER"},
                    "representative_quotes": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "languages_seen": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                },
                "required": ["name", "count", "percentage", "representative_quotes", "languages_seen"],
            },
        },
    },
    "required": ["summary", "sub_issues"],
}


def _handle_analyze_topic(request):
    """Handle POST /analyze-topic — on-demand LLM sub-issue analysis."""
    try:
        body = request.get_json(silent=False)
    except Exception:
        return json.dumps({"error": "invalid JSON body"}), 400, JSON_HEADERS

    topic = (body.get("topic") or "").strip()
    if not topic:
        return json.dumps({"error": "missing required field: topic"}), 400, JSON_HEADERS

    # Accept reviews directly from the frontend (already filtered)
    inline_reviews = body.get("reviews")
    if inline_reviews and isinstance(inline_reviews, list) and len(inline_reviews) > 0:
        reviews = [
            {
                "review_body": (r.get("review_body") or "").strip(),
                "language": r.get("language") or "unknown",
                "rating": r.get("rating"),
                "game": r.get("game") or "unknown",
            }
            for r in inline_reviews
            if (r.get("review_body") or "").strip()
        ]
    else:
        # Fallback: query Firestore directly
        game = (body.get("game") or "").strip() or None
        language = (body.get("language") or "").strip() or None
        try:
            reviews = _fs_query_reviews_by_topic(topic, game, language)
        except Exception as e:
            _log_event("topic_analysis_query_failed", {"topic": topic, "err": str(e)})
            return json.dumps({"error": f"Failed to query reviews: {e}"}), 500, JSON_HEADERS

    if not reviews:
        return json.dumps({"error": "No reviews found for this topic", "topic": topic}), 404, JSON_HEADERS

    total_count = body.get("total_count") or len(reviews)

    # Call LLM
    prompt = _build_topic_analysis_prompt(topic, reviews)
    try:
        llm_result = _gen_json(
            model=VERTEX_MODEL_STAGE2,
            prompt=prompt,
            max_tokens=2048,
            response_schema=TOPIC_ANALYSIS_SCHEMA,
            retries=STAGE2_RETRIES,
        )
    except Exception as e:
        _log_event("topic_analysis_llm_failed", {"topic": topic, "err": str(e)})
        return json.dumps({"error": f"LLM analysis failed: {e}"}), 500, JSON_HEADERS

    now = _utc_iso_now()

    result = {
        "topic": topic,
        "review_count": len(reviews),
        "total_count": total_count,
        "summary": llm_result.get("summary", ""),
        "sub_issues": llm_result.get("sub_issues", []),
        "generated_at": now,
    }

    _log_event("topic_analysis_done", {"topic": topic, "review_count": len(reviews), "total_count": total_count, "sub_issues": len(result["sub_issues"])})
    return json.dumps(result, ensure_ascii=False), 200, JSON_HEADERS


# -----------------------------
# EWMA Forecast Optimization
# -----------------------------

def _fs_write_ewma_config(game: str, config: Dict[str, Any]) -> None:
    """Write optimization config to ewma_config/{game}."""
    _, project = _fs_init_auth()
    token = _fs_get_token()
    url = _fs_doc_url(project, FIRESTORE_DATABASE, EWMA_CONFIG_COLLECTION, game)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    body = {"fields": {k: _fs_value(v) for k, v in config.items()}}
    resp = _HTTP.patch(url, headers=headers, data=json.dumps(body), timeout=FIRESTORE_WRITE_TIMEOUT_SEC)
    if 200 <= resp.status_code < 300:
        _log_event("ewma_config_write_ok", {"game": game})
    else:
        _log_event("ewma_config_write_failed", {"game": game, "http_status": resp.status_code, "resp": _truncate(resp.text, 400)})


def _fs_read_ewma_config(game: str) -> Optional[Dict[str, Any]]:
    """Read optimization config from ewma_config/{game}."""
    try:
        _, project = _fs_init_auth()
        token = _fs_get_token()
        url = _fs_doc_url(project, FIRESTORE_DATABASE, EWMA_CONFIG_COLLECTION, game)
        headers = {"Authorization": f"Bearer {token}"}
        resp = _HTTP.get(url, headers=headers, timeout=FIRESTORE_WRITE_TIMEOUT_SEC)
        if resp.status_code != 200:
            return None
        doc = resp.json()
        fields = doc.get("fields", {})
        return {k: _fs_parse_value(v) for k, v in fields.items()}
    except Exception as e:
        _log_event("ewma_config_read_failed", {"game": game, "err": str(e)})
        return None


def _fs_write_ewma_daily(game: str, date_str: str, record: Dict[str, Any]) -> None:
    """Write a single daily data point to ewma_daily_data/{game}__{date}."""
    _, project = _fs_init_auth()
    token = _fs_get_token()
    doc_id = f"{game}__{date_str}"
    url = _fs_doc_url(project, FIRESTORE_DATABASE, EWMA_DAILY_COLLECTION, doc_id)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    body = {"fields": {k: _fs_value(v) for k, v in record.items()}}
    resp = _HTTP.patch(url, headers=headers, data=json.dumps(body), timeout=FIRESTORE_WRITE_TIMEOUT_SEC)
    if resp.status_code >= 400:
        _log_event("ewma_daily_write_failed", {"game": game, "date": date_str, "status": resp.status_code, "body": resp.text[:200]})


def _fs_list_collection_docs(collection_id: str, game_prefix: str) -> List[Dict[str, Any]]:
    """List all documents in a collection, filtering by doc-ID prefix '{game}__'.

    Uses the simple list-documents API (no runQuery, no indexes needed).
    Paginates automatically.
    """
    _, project = _fs_init_auth()
    token = _fs_get_token()
    base = f"https://firestore.googleapis.com/v1/projects/{project}/databases/{FIRESTORE_DATABASE}/documents"
    headers = {"Authorization": f"Bearer {token}"}

    prefix = f"{game_prefix}__"
    results: List[Dict[str, Any]] = []
    page_token: Optional[str] = None

    while True:
        list_url = f"{base}/{collection_id}?pageSize=300"
        if page_token:
            list_url += f"&pageToken={page_token}"
        resp = _HTTP.get(list_url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for doc in data.get("documents", []):
            # doc["name"] ends with "/{collectionId}/{docId}"
            doc_name = doc.get("name", "")
            doc_id = doc_name.rsplit("/", 1)[-1] if "/" in doc_name else ""
            if not doc_id.startswith(prefix):
                continue
            fields = doc.get("fields", {})
            results.append({k: _fs_parse_value(v) for k, v in fields.items()})
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return results


def _fs_read_ewma_daily_data(game: str) -> List[Dict[str, Any]]:
    """Read ewma_daily_data for a game, ordered by date."""
    docs = _fs_list_collection_docs(EWMA_DAILY_COLLECTION, game)
    results = []
    for parsed in docs:
        results.append({
            "date": parsed.get("date", ""),
            "avg_rating": parsed.get("avg_rating", 0.0),
            "displayed_rating": parsed.get("displayed_rating"),
            "count": parsed.get("count", 0),
        })
    results.sort(key=lambda r: r["date"])
    return results


def _fs_write_ewma_upload_log(game: str, log_entry: Dict[str, Any]) -> None:
    """Write an upload log entry to ewma_upload_log/{game}__{timestamp}."""
    _, project = _fs_init_auth()
    token = _fs_get_token()
    doc_id = f"{game}__{_now_ms()}"
    url = _fs_doc_url(project, FIRESTORE_DATABASE, EWMA_UPLOAD_LOG_COLLECTION, doc_id)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    body = {"fields": {k: _fs_value(v) for k, v in log_entry.items()}}
    resp = _HTTP.patch(url, headers=headers, data=json.dumps(body), timeout=FIRESTORE_WRITE_TIMEOUT_SEC)
    if 200 <= resp.status_code < 300:
        _log_event("ewma_upload_log_write_ok", {"game": game, "doc_id": doc_id})
    else:
        _log_event("ewma_upload_log_write_failed", {"game": game, "http_status": resp.status_code})


def _fs_write_ewma_opt_history(game: str, config: Dict[str, Any]) -> None:
    """Write an optimization history entry to ewma_optimization_history/{game}__{timestamp}."""
    _, project = _fs_init_auth()
    token = _fs_get_token()
    doc_id = f"{game}__{_now_ms()}"
    url = _fs_doc_url(project, FIRESTORE_DATABASE, EWMA_OPT_HISTORY_COLLECTION, doc_id)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    body = {"fields": {k: _fs_value(v) for k, v in config.items()}}
    resp = _HTTP.patch(url, headers=headers, data=json.dumps(body), timeout=FIRESTORE_WRITE_TIMEOUT_SEC)
    if 200 <= resp.status_code < 300:
        _log_event("ewma_opt_history_write_ok", {"game": game, "doc_id": doc_id})
    else:
        _log_event("ewma_opt_history_write_failed", {"game": game, "http_status": resp.status_code})


def _fs_query_collection(collection_id: str, game: str, order_field: str, limit_count: int) -> List[Dict[str, Any]]:
    """List documents filtered by game prefix, sort descending, limit results.

    Uses list-documents API (no runQuery, no indexes needed).
    """
    docs = _fs_list_collection_docs(collection_id, game)
    docs.sort(key=lambda r: r.get(order_field, ""), reverse=True)
    return docs[:limit_count]


def _optimize_alpha(daily_data: List[Dict[str, Any]], game: str) -> Dict[str, Any]:
    """Grid search walk-forward optimization for EWMA alpha. Pure function."""
    n = len(daily_data)
    if n < EWMA_MIN_DAYS_OPTIMIZE:
        return {"status": "insufficient_data", "data_days": n, "min_required": EWMA_MIN_DAYS_OPTIMIZE}

    split_idx = int(n * EWMA_TRAIN_RATIO)
    train_data = daily_data[:split_idx]
    test_data = daily_data[split_idx:]

    train_days = len(train_data)
    test_days = len(test_data)

    # Generate alpha candidates
    alphas = []
    a = EWMA_ALPHA_RANGE_START
    while a <= EWMA_ALPHA_RANGE_END + 1e-9:
        alphas.append(round(a, 4))
        a += EWMA_ALPHA_STEP

    # Seed EWMA from displayed_rating when available (better starting point)
    seed = daily_data[0].get("displayed_rating") or daily_data[0]["avg_rating"]

    best_alpha = EWMA_DEFAULT_ALPHA
    best_mae = float("inf")

    for alpha in alphas:
        one_minus = 1.0 - alpha
        ewma = seed
        errors = []
        for i, d in enumerate(daily_data):
            if i > 0:
                ewma = alpha * d["avg_rating"] + one_minus * ewma
            # Only evaluate on test set, skip burn-in
            if i >= split_idx and i >= EWMA_BURN_IN_DAYS:
                target = d.get("displayed_rating") or d["avg_rating"]
                errors.append(abs(ewma - target))

        if errors:
            mae = sum(errors) / len(errors)
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha

    # Compute optimized alpha's full metrics on test set
    one_minus = 1.0 - best_alpha
    ewma = seed
    test_errors = []
    test_sq_errors = []
    within_005 = 0
    for i, d in enumerate(daily_data):
        if i > 0:
            ewma = best_alpha * d["avg_rating"] + one_minus * ewma
        if i >= split_idx and i >= EWMA_BURN_IN_DAYS:
            target = d.get("displayed_rating") or d["avg_rating"]
            err = abs(ewma - target)
            test_errors.append(err)
            test_sq_errors.append(err * err)
            if err <= 0.05:
                within_005 += 1

    mae = sum(test_errors) / len(test_errors) if test_errors else 0.0
    rmse = math.sqrt(sum(test_sq_errors) / len(test_sq_errors)) if test_sq_errors else 0.0
    pct_within = (within_005 / len(test_errors) * 100) if test_errors else 0.0

    # Compute default alpha's MAE for comparison
    one_minus_def = 1.0 - EWMA_DEFAULT_ALPHA
    ewma_def = seed
    def_errors = []
    for i, d in enumerate(daily_data):
        if i > 0:
            ewma_def = EWMA_DEFAULT_ALPHA * d["avg_rating"] + one_minus_def * ewma_def
        if i >= split_idx and i >= EWMA_BURN_IN_DAYS:
            target = d.get("displayed_rating") or d["avg_rating"]
            def_errors.append(abs(ewma_def - target))

    default_mae = sum(def_errors) / len(def_errors) if def_errors else 0.0
    improvement_pct = ((default_mae - mae) / default_mae * 100) if default_mae > 0 else 0.0

    now = _utc_iso_now()
    config = {
        "status": "optimized",
        "game": game,
        "optimized_alpha": round(best_alpha, 4),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "pct_within_005": round(pct_within, 2),
        "data_days": n,
        "train_days": train_days,
        "test_days": test_days,
        "default_alpha": EWMA_DEFAULT_ALPHA,
        "default_mae": round(default_mae, 6),
        "improvement_pct": round(improvement_pct, 2),
        "last_optimized_at": now,
    }
    return config


def _detect_csv_columns(headers: List[str]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Auto-detect Date, Daily Average Rating, and Total Average Rating columns from stats_ratings CSV.

    Returns (date_idx, daily_avg_idx, total_avg_idx).
    """
    date_idx = None
    daily_avg_idx = None
    total_avg_idx = None
    for i, h in enumerate(headers):
        h_lower = h.strip().lower()
        if date_idx is None and h_lower == "date":
            date_idx = i
        elif daily_avg_idx is None and "daily" in h_lower and "average" in h_lower and "rating" in h_lower:
            daily_avg_idx = i
        elif total_avg_idx is None and "total" in h_lower and "average" in h_lower and "rating" in h_lower:
            total_avg_idx = i
    return date_idx, daily_avg_idx, total_avg_idx


def _handle_upload_ewma_csv(request):
    """POST /upload-ewma-csv — parse stats_ratings CSV, store in ewma_daily_data.

    Expects Google Play Console stats_ratings CSV (UTF-16 LE with BOM) with columns:
    Date, Package Name, Daily Average Rating, Total Average Rating
    """
    try:
        game = (request.form.get("game") or "").strip().upper()
        if game not in VALID_GAMES:
            return json.dumps({"error": f"Invalid game. Must be one of: {sorted(VALID_GAMES)}"}), 400, JSON_HEADERS

        uploaded_file = request.files.get("file")
        if not uploaded_file:
            return json.dumps({"error": "No file uploaded. Send a CSV file in the 'file' field."}), 400, JSON_HEADERS

        raw = uploaded_file.read()

        # Stats CSV is UTF-16 LE with BOM; fall back to UTF-8-sig
        try:
            text = raw.decode("utf-16")
        except (UnicodeDecodeError, UnicodeError):
            text = raw.decode("utf-8-sig", errors="replace")

        reader = csv.reader(io.StringIO(text))

        header_row = next(reader, None)
        if not header_row:
            return json.dumps({"error": "CSV is empty or has no header row."}), 400, JSON_HEADERS

        # Strip whitespace from headers (UTF-16 export may have padding)
        header_row = [h.strip() for h in header_row]

        date_idx, daily_avg_idx, total_avg_idx = _detect_csv_columns(header_row)
        if date_idx is None or daily_avg_idx is None or total_avg_idx is None:
            return json.dumps({
                "error": "Could not detect stats_ratings columns.",
                "headers_found": header_row,
                "hint": "Need columns: 'Date', 'Daily Average Rating', 'Total Average Rating'.",
            }), 400, JSON_HEADERS

        max_idx = max(date_idx, daily_avg_idx, total_avg_idx)

        # Each row is already a daily aggregate — no aggregation needed
        daily: Dict[str, Dict[str, Any]] = {}
        for row in reader:
            if len(row) <= max_idx:
                continue
            raw_date = row[date_idx].strip()
            raw_daily_avg = row[daily_avg_idx].strip()
            raw_total_avg = row[total_avg_idx].strip()

            # Parse date (YYYY-MM-DD)
            date_str = raw_date[:10]
            try:
                datetime.datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue

            # Parse rating values
            try:
                avg_rating = float(raw_daily_avg)
                displayed_rating = float(raw_total_avg)
            except (ValueError, TypeError):
                continue

            daily[date_str] = {
                "avg_rating": round(avg_rating, 4),
                "displayed_rating": round(displayed_rating, 4),
            }

        if not daily:
            return json.dumps({"error": "No valid data rows found in CSV."}), 400, JSON_HEADERS

        # Write each day to Firestore
        now = _utc_iso_now()
        for date_str, vals in daily.items():
            record = {
                "game": game,
                "date": date_str,
                "avg_rating": vals["avg_rating"],
                "displayed_rating": vals["displayed_rating"],
                "source": "stats_csv",
                "uploaded_at": now,
            }
            _fs_write_ewma_daily(game, date_str, record)

        dates_sorted = sorted(daily.keys())
        result = {
            "status": "ok",
            "game": game,
            "days_uploaded": len(daily),
            "date_range": {"min": dates_sorted[0], "max": dates_sorted[-1]},
        }

        # Write upload log entry
        log_entry = {
            "game": game,
            "filename": uploaded_file.filename or "unknown",
            "days_uploaded": len(daily),
            "date_range_min": dates_sorted[0],
            "date_range_max": dates_sorted[-1],
            "uploaded_at": now,
        }
        try:
            _fs_write_ewma_upload_log(game, log_entry)
        except Exception as log_err:
            _log_event("ewma_upload_log_write_error", {"game": game, "err": str(log_err)})

        _log_event("ewma_csv_uploaded", result)
        return json.dumps(result, ensure_ascii=False), 200, JSON_HEADERS

    except Exception as e:
        _log_event("ewma_csv_upload_failed", {"err": str(e)})
        return json.dumps({"error": f"Upload failed: {e}"}), 500, JSON_HEADERS


def _handle_optimize_ewma(request):
    """GET /optimize-ewma?game=GI — run grid search, write result to ewma_config/{game}."""
    game = (request.args.get("game") or "").strip().upper()
    if game not in VALID_GAMES:
        return json.dumps({"error": f"Invalid game. Must be one of: {sorted(VALID_GAMES)}"}), 400, JSON_HEADERS

    try:
        daily_data = _fs_read_ewma_daily_data(game)
        if not daily_data:
            return json.dumps({"error": "No daily data found for this game. Upload a CSV first."}), 404, JSON_HEADERS

        config = _optimize_alpha(daily_data, game)

        if config.get("status") == "optimized":
            _fs_write_ewma_config(game, config)
            # Write optimization history entry
            try:
                _fs_write_ewma_opt_history(game, config)
            except Exception as hist_err:
                _log_event("ewma_opt_history_write_error", {"game": game, "err": str(hist_err)})

        _log_event("ewma_optimization_done", {"game": game, "status": config.get("status"), "alpha": config.get("optimized_alpha")})
        return json.dumps(config, ensure_ascii=False), 200, JSON_HEADERS

    except Exception as e:
        _log_event("ewma_optimization_failed", {"game": game, "err": str(e)})
        return json.dumps({"error": f"Optimization failed: {e}"}), 500, JSON_HEADERS


def _handle_ewma_daily_data(request):
    """GET /ewma-daily-data?game=GI — return daily data + cached optimization config."""
    game = (request.args.get("game") or "").strip().upper()
    if game not in VALID_GAMES:
        return json.dumps({"error": f"Invalid game. Must be one of: {sorted(VALID_GAMES)}"}), 400, JSON_HEADERS

    try:
        daily_data = _fs_read_ewma_daily_data(game)
        ewma_config = _fs_read_ewma_config(game)

        result = {
            "game": game,
            "daily_data": daily_data,
            "ewma_config": ewma_config,
        }
        return json.dumps(result, ensure_ascii=False), 200, JSON_HEADERS

    except Exception as e:
        _log_event("ewma_daily_data_read_failed", {"game": game, "err": str(e)})
        return json.dumps({"error": f"Failed to read data: {e}"}), 500, JSON_HEADERS


def _handle_ewma_upload_log(request):
    """GET /ewma-upload-log?game=GI — return recent upload log entries."""
    game = (request.args.get("game") or "").strip().upper()
    if game not in VALID_GAMES:
        return json.dumps({"error": f"Invalid game. Must be one of: {sorted(VALID_GAMES)}"}), 400, JSON_HEADERS

    try:
        uploads = _fs_query_collection(EWMA_UPLOAD_LOG_COLLECTION, game, "uploaded_at", 50)
        return json.dumps({"game": game, "uploads": uploads}, ensure_ascii=False), 200, JSON_HEADERS
    except Exception as e:
        _log_event("ewma_upload_log_read_failed", {"game": game, "err": str(e)})
        return json.dumps({"error": f"Failed to read upload log: {e}"}), 500, JSON_HEADERS


def _handle_ewma_opt_history(request):
    """GET /ewma-opt-history?game=GI — return optimization history entries."""
    game = (request.args.get("game") or "").strip().upper()
    if game not in VALID_GAMES:
        return json.dumps({"error": f"Invalid game. Must be one of: {sorted(VALID_GAMES)}"}), 400, JSON_HEADERS

    try:
        history = _fs_query_collection(EWMA_OPT_HISTORY_COLLECTION, game, "last_optimized_at", 20)
        return json.dumps({"game": game, "history": history}, ensure_ascii=False), 200, JSON_HEADERS
    except Exception as e:
        _log_event("ewma_opt_history_read_failed", {"game": game, "err": str(e)})
        return json.dumps({"error": f"Failed to read optimization history: {e}"}), 500, JSON_HEADERS


# -----------------------------
# Async helpers
# -----------------------------
def _enqueue_review_task(event_id: str, payload: Dict[str, Any]) -> bool:
    """Enqueue a review for async processing via Cloud Tasks.
    Returns True if enqueued successfully, False otherwise."""
    try:
        from google.cloud import tasks_v2

        client = _get_tasks_client()
        project = _project_id()

        queue_path = client.queue_path(project, CLOUD_TASKS_LOCATION, CLOUD_TASKS_QUEUE)

        service_url = CLOUD_RUN_SERVICE_URL
        if not service_url:
            _log_event("enqueue_missing_service_url", {"event_id": event_id})
            return False

        target_url = f"{service_url.rstrip('/')}/internal/process"

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if INTERNAL_PROCESS_TOKEN:
            headers["X-Internal-Process-Token"] = INTERNAL_PROCESS_TOKEN

        task_config = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": target_url,
                "headers": headers,
                "body": body,
            }
        }

        if CLOUD_TASKS_SERVICE_ACCOUNT:
            task_config["http_request"]["oidc_token"] = {
                "service_account_email": CLOUD_TASKS_SERVICE_ACCOUNT,
                "audience": service_url,
            }

        safe_task_name = re.sub(r"[^A-Za-z0-9_-]", "_", event_id)[:500]
        task_config["name"] = f"{queue_path}/tasks/{safe_task_name}"

        response = client.create_task(parent=queue_path, task=task_config)

        _log_event("task_enqueued", {
            "event_id": event_id,
            "task_name": response.name,
            "target_url": target_url,
        })
        return True

    except Exception as e:
        _log_event("task_enqueue_failed", {
            "event_id": event_id,
            "err": str(e),
        })
        return False


def _call_csc_callback(event_id: str, result: Dict[str, Any]) -> bool:
    """Call CSC callback API to deliver the processing result.
    Returns True if callback succeeded (or is disabled), False otherwise."""
    if not CSC_CALLBACK_URL:
        _log_event("csc_callback_skipped", {"event_id": event_id, "reason": "no_callback_url"})
        return True

    try:
        headers = {"Content-Type": "application/json"}
        if CSC_CALLBACK_TOKEN and CSC_CALLBACK_TOKEN_HEADER:
            headers[CSC_CALLBACK_TOKEN_HEADER] = CSC_CALLBACK_TOKEN

        resp = _HTTP.post(
            CSC_CALLBACK_URL,
            headers=headers,
            data=json.dumps(result, ensure_ascii=False),
            timeout=CSC_CALLBACK_TIMEOUT_SEC,
        )

        if not (200 <= resp.status_code < 300):
            _log_event("csc_callback_failed", {
                "event_id": event_id,
                "status": resp.status_code,
                "resp_snippet": _truncate(resp.text, 500),
            })
            return False

        try:
            resp_body = resp.json()
            retcode = resp_body.get("retcode")
            if retcode is not None and retcode != 0:
                _log_event("csc_callback_failed", {
                    "event_id": event_id,
                    "status": resp.status_code,
                    "retcode": retcode,
                    "message": resp_body.get("message", ""),
                })
                return False
        except (ValueError, AttributeError):
            pass

        _log_event("csc_callback_ok", {
            "event_id": event_id,
            "status": resp.status_code,
        })
        return True

    except Exception as e:
        _log_event("csc_callback_failed", {
            "event_id": event_id,
            "err": str(e),
        })
        return False


def _process_review(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Run the full review processing pipeline (Stage 1 + Stage 2 + template).
    Returns (result_dict, is_retryable_error)."""
    start_ms = _now_ms()

    order_id = payload.get("order_id")
    event_id = payload.get("event_id") or f"evt_{order_id}_{int(time.time())}"
    rating = payload.get("rating")
    game_biz = payload.get("game_biz") or ""
    lang_code = payload.get("language") or ""
    title = (payload.get("title") or "").strip()
    body = (payload.get("body") or "").strip()

    # --- Stage 1: Sensitive Content Classification ---
    stage1_prompt = _stage1_prompt(payload, title, body)
    stage1_schema = _stage1_response_schema()

    try:
        stage1 = _call_llm_with_fallback(
            model=VERTEX_MODEL_STAGE1,
            prompt=stage1_prompt,
            schema=stage1_schema,
            validator=_validate_stage1,
            retries=STAGE1_RETRIES,
            stage_name="stage1",
            event_id=event_id,
            order_id=order_id,
        )
    except Exception as e:
        _log_event("stage1_llm_failed", {"event_id": event_id, "order_id": order_id, "err": str(e)})
        result = _build_result(
            event_id=event_id, order_id=order_id, action="NEEDS_HUMAN",
            payload=payload, error="stage1_llm_failed",
        )
        latency = _now_ms() - start_ms
        _log_event("decision", {**result, "latency_ms": latency})
        _firestore_write_review_event_best_effort(event_id=event_id, payload=payload, result=result, latency_ms=latency)
        return result, True  # retryable

    stage1_bucket = stage1.get("bucket")
    if stage1_bucket == "UNCERTAIN":
        action = "NEEDS_HUMAN"
    elif stage1_bucket != "NONE":
        action = "TAG_AND_CLOSE"
    else:
        action = "STAGE2"

    if action in ("TAG_AND_CLOSE", "NEEDS_HUMAN"):
        result = _build_result(
            event_id=event_id, order_id=order_id, action=action,
            payload=payload, stage1=stage1,
        )
        latency = _now_ms() - start_ms
        _log_event("decision", {**result, "latency_ms": latency})
        _firestore_write_review_event_best_effort(event_id=event_id, payload=payload, result=result, latency_ms=latency)
        return result, False

    # --- Stage 2 ---
    game = GAME_BIZ_TO_GAME.get(game_biz)
    if not game:
        result = _build_result(
            event_id=event_id, order_id=order_id, action="NEEDS_HUMAN",
            payload=payload, stage1=stage1, reason="unknown_game_biz",
        )
        latency = _now_ms() - start_ms
        _log_event("decision", {**result, "latency_ms": latency})
        _firestore_write_review_event_best_effort(event_id=event_id, payload=payload, result=result, latency_ms=latency)
        return result, False

    allowed_topics = TOPICS_BY_GAME.get(game, [])
    if not allowed_topics:
        result = _build_result(
            event_id=event_id, order_id=order_id, action="NEEDS_HUMAN",
            payload=payload, stage1=stage1, game=game, reason="no_topics_in_registry_for_game",
        )
        latency = _now_ms() - start_ms
        _log_event("decision", {**result, "latency_ms": latency})
        _firestore_write_review_event_best_effort(event_id=event_id, payload=payload, result=result, latency_ms=latency)
        return result, False

    stage2_prompt = _stage2_prompt(payload, game, allowed_topics, title, body)
    stage2_schema = _stage2_response_schema(allowed_topics)

    try:
        stage2 = _call_llm_with_fallback(
            model=VERTEX_MODEL_STAGE2,
            prompt=stage2_prompt,
            schema=stage2_schema,
            validator=lambda r: _validate_stage2(r, allowed_topics, game),
            retries=STAGE2_RETRIES,
            stage_name="stage2",
            event_id=event_id,
            order_id=order_id,
        )
    except Exception as e2:
        _log_event("stage2_llm_failed", {"event_id": event_id, "order_id": order_id, "err": str(e2)})
        result = _build_result(
            event_id=event_id, order_id=order_id, action="NEEDS_HUMAN",
            payload=payload, stage1=stage1, error="stage2_llm_failed",
        )
        latency = _now_ms() - start_ms
        _log_event("decision", {**result, "latency_ms": latency})
        _firestore_write_review_event_best_effort(event_id=event_id, payload=payload, result=result, latency_ms=latency)
        return result, True  # retryable

    stage2_conf = float(stage2.get("confidence") or 0.0)
    if stage2_conf < STAGE2_CONFIDENCE_THRESHOLD:
        result = _build_result(
            event_id=event_id, order_id=order_id, action="NEEDS_HUMAN",
            payload=payload, stage1=stage1, stage2=stage2, game=game, reason="stage2_low_confidence",
        )
        latency = _now_ms() - start_ms
        _log_event("decision", {**result, "latency_ms": latency})
        _firestore_write_review_event_best_effort(event_id=event_id, payload=payload, result=result, latency_ms=latency)
        return result, False

    issue_type = (stage2.get("issue_type") or "").strip()
    topic = (stage2.get("topic") or "").strip()

    lang_key, lang_fallback = _lang_to_template_key(lang_code)
    tpl, tpl_lang_fallback = _select_template(game, issue_type, topic, lang_key)

    if not tpl:
        result = _build_result(
            event_id=event_id, order_id=order_id, action="NEEDS_HUMAN",
            payload=payload, stage1=stage1, stage2=stage2, game=game,
            reason="template_not_found", lang_key=lang_key,
        )
        latency = _now_ms() - start_ms
        _log_event("decision", {**result, "latency_ms": latency})
        _firestore_write_review_event_best_effort(event_id=event_id, payload=payload, result=result, latency_ms=latency)
        return result, False

    template_id = tpl.get("template_id")
    reply_text = tpl.get("template_text") or ""

    max_chars = _max_reply_chars()
    template_error = _validate_template_length(reply_text, max_chars)
    if template_error:
        result = _build_result(
            event_id=event_id, order_id=order_id, action="NEEDS_HUMAN",
            payload=payload, stage1=stage1, stage2=stage2, game=game,
            reason=template_error, template_id=template_id,
            max_chars=max_chars, actual_chars=_count_runes(reply_text),
        )
        latency = _now_ms() - start_ms
        _log_event("decision", {**result, "latency_ms": latency})
        _firestore_write_review_event_best_effort(event_id=event_id, payload=payload, result=result, latency_ms=latency)
        return result, False

    result = _build_result(
        event_id=event_id, order_id=order_id, action="REPLY_AND_CLOSE",
        payload=payload, stage1=stage1, stage2=stage2, game=game,
        template={
            "template_id": template_id,
            "language_key": lang_key,
            "lang_fallback_used": (lang_fallback or tpl_lang_fallback),
            "reply_text": reply_text,
        },
    )
    latency = _now_ms() - start_ms
    _log_event("decision", {**result, "latency_ms": latency})
    _firestore_write_review_event_best_effort(event_id=event_id, payload=payload, result=result, latency_ms=latency)
    return result, False


def _handle_get_result(request):
    """Return async processing result for a given event_id from Firestore."""
    parts = request.path.split("/results/", 1)
    raw_event_id = parts[1] if len(parts) > 1 else ""
    if not raw_event_id:
        return json.dumps({"status": "error", "error": "missing event_id"}), 400, JSON_HEADERS

    try:
        _, project = _fs_init_auth()
        token = _fs_get_token()
    except Exception as e:
        return json.dumps({"status": "error", "error": f"auth: {e}"}), 500, JSON_HEADERS

    doc_id = _fs_safe_doc_id(raw_event_id)
    url = _fs_doc_url(project, FIRESTORE_DATABASE, FIRESTORE_COLLECTION, doc_id)
    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = _HTTP.get(url, headers=headers, timeout=FIRESTORE_WRITE_TIMEOUT_SEC)
    except Exception as e:
        return json.dumps({"status": "error", "error": f"firestore request failed: {e}"}), 502, JSON_HEADERS

    if resp.status_code == 404:
        return json.dumps({"status": "pending"}), 200, JSON_HEADERS

    if resp.status_code != 200:
        return json.dumps({"status": "error", "error": f"firestore HTTP {resp.status_code}"}), 502, JSON_HEADERS

    try:
        doc = resp.json()
        fields = doc.get("fields", {})
        parsed = {k: _fs_parse_value(v) for k, v in fields.items()}
        parsed["status"] = "done"
        return json.dumps(parsed, ensure_ascii=False), 200, JSON_HEADERS
    except Exception as e:
        return json.dumps({"status": "error", "error": f"parse: {e}"}), 500, JSON_HEADERS


def _handle_internal_process(request):
    """Handle async processing triggered by Cloud Tasks."""
    if not INTERNAL_PROCESS_TOKEN:
        _log_event("internal_auth_skipped_no_token", {"path": request.path})
    else:
        got = request.headers.get("X-Internal-Process-Token", "")
        if not secrets.compare_digest(got, INTERNAL_PROCESS_TOKEN):
            _log_event("internal_process_auth_failed", {"path": request.path})
            return "unauthorized", 401

    try:
        payload = request.get_json(silent=False)
    except Exception:
        return "invalid json", 400

    event_id = payload.get("event_id") or "unknown"
    _log_event("internal_process_start", {"event_id": event_id})

    # Check Firestore for existing result to avoid re-running LLM on retries
    cached_result = _firestore_read_existing_result(event_id)
    if cached_result:
        _log_event("internal_process_cache_hit", {"event_id": event_id, "action": cached_result.get("action")})
        callback_ok = _call_csc_callback(event_id, cached_result)
        if not callback_ok:
            _log_event("internal_process_callback_failed", {"event_id": event_id})
            return json.dumps({"error": "callback_failed", "event_id": event_id}), 500, JSON_HEADERS
        _log_event("internal_process_done", {"event_id": event_id, "action": cached_result.get("action")})
        return json.dumps(cached_result, ensure_ascii=False), 200, JSON_HEADERS

    result, is_retryable = _process_review(payload)

    if is_retryable:
        _log_event("internal_process_retryable_failure", {
            "event_id": event_id,
            "action": result.get("action"),
        })
        return json.dumps({"error": "retryable_failure", "event_id": event_id}), 500, JSON_HEADERS

    callback_ok = _call_csc_callback(event_id, result)

    if not callback_ok:
        _log_event("internal_process_callback_failed", {"event_id": event_id})
        return json.dumps({"error": "callback_failed", "event_id": event_id}), 500, JSON_HEADERS

    _log_event("internal_process_done", {"event_id": event_id, "action": result.get("action")})
    return json.dumps(result, ensure_ascii=False), 200, JSON_HEADERS


# -----------------------------
# HTTP handler
# -----------------------------
@functions_framework.http
def review_webhook(request):
    start_ms = _now_ms()

    if request.path in ("/healthz", "/health", "/") and request.method == "GET":
        return "ok", 200

    # Route: fetch async processing result by event_id
    if request.path.startswith("/results/") and request.method == "GET":
        return _handle_get_result(request)

    # Route: internal async processing endpoint (called by Cloud Tasks)
    if request.path == "/internal/process" and request.method == "POST":
        return _handle_internal_process(request)

    # Auth gate for tool/data endpoints
    if request.path in ("/analyze-topic", "/upload-ewma-csv", "/optimize-ewma",
                         "/ewma-daily-data", "/ewma-upload-log", "/ewma-opt-history"):
        if not WEBHOOK_TOKEN:
            _log_event("tool_auth_skipped_no_token", {"path": request.path})
        else:
            got = request.headers.get(WEBHOOK_HEADER, "")
            if not secrets.compare_digest(got, WEBHOOK_TOKEN):
                _log_event("tool_auth_failed", {"path": request.path})
                return "unauthorized", 401

    # Route: on-demand topic sub-issue analysis
    if request.path == "/analyze-topic" and request.method == "POST":
        return _handle_analyze_topic(request)

    # Routes: EWMA Forecast Optimization
    if request.path == "/upload-ewma-csv" and request.method == "POST":
        return _handle_upload_ewma_csv(request)

    if request.path == "/optimize-ewma" and request.method == "GET":
        return _handle_optimize_ewma(request)

    if request.path == "/ewma-daily-data" and request.method == "GET":
        return _handle_ewma_daily_data(request)

    if request.path == "/ewma-upload-log" and request.method == "GET":
        return _handle_ewma_upload_log(request)

    if request.path == "/ewma-opt-history" and request.method == "GET":
        return _handle_ewma_opt_history(request)

    if not WEBHOOK_TOKEN:
        _log_event("webhook_auth_skipped_no_token", {"path": request.path})
    else:
        got = request.headers.get(WEBHOOK_HEADER, "")
        if not secrets.compare_digest(got, WEBHOOK_TOKEN):
            _log_event(
                "webhook_auth_failed",
                {"path": request.path, "header": WEBHOOK_HEADER},
            )
            return "unauthorized", 401

    try:
        payload = request.get_json(silent=False)
    except Exception:
        return "invalid json", 400

    order_id = payload.get("order_id")
    event_id = payload.get("event_id") or f"evt_{order_id}_{int(time.time())}"
    rating = payload.get("rating")
    game_biz = payload.get("game_biz") or ""

    if order_id is None or rating is None or not game_biz:
        _log_event("webhook_missing_fields", {"event_id": event_id, "order_id": order_id})
        return "missing required fields", 400

    # Ensure event_id is in payload for downstream processing
    payload["event_id"] = event_id

    # -----------------------------
    # Rule-based Bad Review Gate
    # -----------------------------
    title = (payload.get("title") or "").strip()
    body = (payload.get("body") or "").strip()
    has_text = bool(title or body)
    rating_eligible = 1 <= rating <= BAD_RATING_MAX

    gate_result = "ELIGIBLE"
    gate_reason = None

    if not has_text and REQUIRES_TEXT:
        gate_result = "NOOP"
        gate_reason = "no_text"
    elif not rating_eligible:
        gate_result = "NOOP"
        gate_reason = "rating_out_of_range"

    if gate_result == "NOOP":
        _log_event("gate_noop", {"event_id": event_id, "order_id": order_id, "gate_reason": gate_reason, "rating": rating, "has_text": has_text})
        return _respond(
            start_ms=start_ms,
            event_id=event_id,
            order_id=order_id,
            action="NOOP",
            payload=payload,
            gate_result=gate_result,
            gate_reason=gate_reason,
        )

    # -----------------------------
    # Async branch: enqueue to Cloud Tasks
    # -----------------------------
    if ASYNC_ENABLED:
        enqueued = _enqueue_review_task(event_id, payload)
        if enqueued:
            accepted_response = {
                "status": "accepted",
                "event_id": event_id,
                "order_id": order_id,
                "processing_mode": "async",
            }
            _log_event("webhook_accepted_async", {"event_id": event_id, "order_id": order_id})
            return json.dumps(accepted_response, ensure_ascii=False), 200, JSON_HEADERS
        else:
            # Enqueue failed -- fall back to synchronous processing
            _log_event("async_fallback_to_sync", {"event_id": event_id, "order_id": order_id})

    # -----------------------------
    # Synchronous processing (default, or async fallback)
    # -----------------------------
    result, _ = _process_review(payload)
    return json.dumps(result, ensure_ascii=False), 200, JSON_HEADERS
