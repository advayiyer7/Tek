"""Build a balanced general-domain reranking set from MS MARCO (v1.1).

Mixing general data with our narrow domain pairs is the anti-forgetting lever:
the base reranker is strong *because* of MS-MARCO pretraining, so keeping some
of that signal in the fine-tune preserves broad ranking ability while the
domain pairs add Tek-specific knowledge. Training-time only; nothing here ships.
"""

from __future__ import annotations

import json
import sys

from .config import DATA

GENERAL_PATH = DATA / "general.jsonl"
N_QUERIES = 5000   # streamed from MS MARCO train
NEG_PER_POS = 2


def main() -> int:
    from datasets import load_dataset

    ds = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
    n_pos = n_neg = 0
    with GENERAL_PATH.open("w", encoding="utf-8") as out:
        for i, row in enumerate(ds):
            if i >= N_QUERIES:
                break
            q = row["query"]
            texts = row["passages"]["passage_text"]
            sel = row["passages"]["is_selected"]
            pos = [t for t, s in zip(texts, sel) if s == 1]
            neg = [t for t, s in zip(texts, sel) if s == 0]
            if not pos:
                continue
            base = {"anchor_id": f"ms{i:06d}", "path": f"__msmarco__/{i}",
                    "topic": "general", "style": "msmarco", "query": q}
            out.write(json.dumps({**base, "doc": pos[0], "label": 1}) + "\n")
            n_pos += 1
            for t in neg[:NEG_PER_POS]:
                out.write(json.dumps({**base, "doc": t, "label": 0}) + "\n")
                n_neg += 1
            if (i + 1) % 1000 == 0:
                print(f"  {i+1} queries -> {n_pos} pos / {n_neg} neg", flush=True)
    print(f"wrote {GENERAL_PATH.name}: {n_pos} pos / {n_neg} neg", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
