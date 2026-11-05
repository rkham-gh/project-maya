#!/usr/bin/env python3
"""Nemotron 3.5 Super's tokenizer pack for maya-nem: the tokenizer and the chat template its GGUF carries
(tools/strata_tokenizer.py's extract; the `tekken` pre-tokenizer), with its end-of-turn token named the way the server
reads it.

    python tools/nem/make_pack.py --gguf <...-00001-of-0000N.gguf> --out <pack dir>      (writes <pack dir>/tokenizer)

The server (serve/server.py) reads the pack's tokenizer/ (its chat template included) and starts build-nem/maya-nem
with the GGUF itself: maya-nem reads the weights straight from the shards, so the pack holds no weights.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import strata_tokenizer as ST  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the first shard of Nemotron 3.5 Super's GGUF")
    ap.add_argument("--out", required=True, help="the pack directory (tokenizer/ goes inside)")
    a = ap.parse_args()
    tok = pathlib.Path(a.out) / "tokenizer"
    cfg = ST.extract(a.gguf, a.out)   # (writes <out>/tokenizer)
    if cfg["pre"] != "tekken":
        raise SystemExit(f"{a.gguf}: pre-tokenizer {cfg['pre']!r}, not Nemotron 3.5's 'tekken'")
    meta = json.loads((tok / "tokenizer.json").read_text(encoding="utf-8"))
    sid = meta.get("special_ids") or {}
    # (the server ends a request at the "eos" a pack names: the GGUF's, <|im_end|>)
    meta["special_ids"] = {"bos": int(sid.get("tokenizer.ggml.bos_token_id", 1)),
                           "eos": int(sid.get("tokenizer.ggml.eos_token_id", 11)),
                           "pad": int(sid.get("tokenizer.ggml.padding_token_id", 11))}
    (tok / "tokenizer.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if not (tok / "chat_template.jinja").exists():
        raise SystemExit(f"{a.gguf}: no chat template in the GGUF")
    print(f"{tok}: vocab {meta['vocab_size']}, merges {meta['n_merges']}, pre {meta['pre']}, "
          f"specials {meta['special_ids']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
