"""Generate synthetic search queries per anchor with local Ollama (llama3.2).

Four styles per anchor, matching how real users search:
  keyword    — terse, a few salient terms or an ID
  natural    — a full natural-language question
  paraphrase — same intent, deliberately avoiding the doc's wording
  partial    — vague / partial-recall ("that thing about ...")

Fully offline (local Ollama). Resumable: appends to queries.jsonl and skips
anchors already done, so a multi-hour run survives interruption. Falls back to
deterministic templates if Ollama is unreachable or returns unparseable output,
so the pipeline always produces a dataset.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import ANCHORS_PATH, CORPUS_DIR, QUERIES_PATH

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3.2:3b"
STYLES = ("keyword", "natural", "paraphrase", "partial")

PROMPT = """You generate realistic file-search queries. A user has a personal note and wants to FIND it again by searching.

NOTE (excerpt):
\"\"\"{snippet}\"\"\"

The user is trying to recall: {fact} (answer: {value}).

Write exactly four search queries that should retrieve THIS note, one per style. Return ONLY a compact JSON object with these keys:
- "keyword": 2-5 salient words or the literal code/id, no question
- "natural": a full natural-language question
- "paraphrase": the same question worded WITHOUT reusing the note's distinctive words
- "partial": a vague half-remembered phrasing, lowercase, no punctuation

JSON:"""


def _ollama(prompt: str, timeout: float = 60.0) -> str | None:
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # constrain Ollama to emit valid JSON (near-100% parse rate)
        "options": {"temperature": 0.5, "num_predict": 256},
        "keep_alive": "10m",
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()).get("response", "")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def _parse(raw: str) -> dict[str, str] | None:
    if not raw:
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    out = {}
    for s in STYLES:
        v = obj.get(s)
        if isinstance(v, str) and v.strip():
            out[s] = " ".join(v.strip().split())[:160]
    return out or None


def _fallback(fact: str, value: str) -> dict[str, str]:
    """Deterministic templates — used only when the LLM path fails."""
    f = fact.replace("what ", "").replace("when ", "").replace("who ", "").replace("which ", "")
    return {
        "keyword": value if len(value) <= 24 else " ".join(fact.split()[-4:]),
        "natural": fact + "?",
        "paraphrase": "can you remind me " + f,
        "partial": " ".join(fact.split()[-3:]).lower(),
    }


def ollama_alive() -> bool:
    return _ollama("Reply with the single word: ok", timeout=30.0) is not None


def _snippet(path: str) -> str:
    try:
        return (CORPUS_DIR / path).read_text(encoding="utf-8")[:480]
    except OSError:
        return ""


def _gen_one(anchor: dict, use_llm: bool) -> list[dict]:
    parsed = None
    if use_llm:
        raw = _ollama(PROMPT.format(snippet=_snippet(anchor["path"]), fact=anchor["fact"], value=anchor["value"]))
        parsed = _parse(raw or "")
    src = parsed or _fallback(anchor["fact"], anchor["value"])
    # backfill any missing style from the fallback
    fb = _fallback(anchor["fact"], anchor["value"])
    rows = []
    for style in STYLES:
        q = src.get(style) or fb[style]
        rows.append({
            "anchor_id": anchor["id"], "path": anchor["path"], "topic": anchor["topic"],
            "style": style, "query": q, "llm": parsed is not None,
        })
    return rows


def main() -> int:
    anchors = [json.loads(l) for l in ANCHORS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    done: set[str] = set()
    if QUERIES_PATH.exists():
        for l in QUERIES_PATH.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["anchor_id"])
    todo = [a for a in anchors if a["id"] not in done]
    use_llm = ollama_alive()
    print(f"anchors={len(anchors)} done={len(done)} todo={len(todo)} "
          f"llm={'ollama:llama3.2' if use_llm else 'FALLBACK-TEMPLATES'}", flush=True)

    written = 0
    with QUERIES_PATH.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = {pool.submit(_gen_one, a, use_llm): a for a in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                for row in fut.result():
                    out.write(json.dumps(row) + "\n")
                    written += 1
                if i % 50 == 0:
                    out.flush()
                    print(f"  {i}/{len(todo)} anchors ({written} queries)", flush=True)
    print(f"done: wrote {written} queries", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
