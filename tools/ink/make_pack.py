#!/usr/bin/env python3
"""Inkling-Small's tokenizer pack for maya-ink: the tokenizer and the chat template its GGUF carries
(tools/strata_tokenizer.py's extract), with its end-of-turn token named the way the server reads it.

    python tools/ink/make_pack.py --gguf <...-00001-of-0000N.gguf> --out <pack dir>      (writes <pack dir>/tokenizer)

The server (serve/server.py) reads the pack's tokenizer/ (its chat template included) and starts build-ink/maya-ink
with the GGUF itself: maya-ink reads the weights straight from the shards, so the pack holds no weights.
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
    ap.add_argument("--gguf", required=True, help="the first shard of Inkling-Small's GGUF")
    ap.add_argument("--out", required=True, help="the pack directory (tokenizer/ goes inside)")
    a = ap.parse_args()
    tok = pathlib.Path(a.out) / "tokenizer"
    cfg = ST.extract(a.gguf, a.out)   # (extract writes <out>/tokenizer)
    if cfg["pre"] != "inkling":
        raise SystemExit(f"{a.gguf}: pre-tokenizer {cfg['pre']!r}, not Inkling's 'inkling'")
    meta = json.loads((tok / "tokenizer.json").read_text(encoding="utf-8"))
    sid = meta.get("special_ids") or {}
    # (the server ends a request at the "eos" a pack names - maya-ink's own end, <|content_model_end_sampling|>)
    meta["special_ids"] = {"bos": int(sid.get("tokenizer.ggml.bos_token_id", 200006)),
                           "eos": int(sid.get("tokenizer.ggml.eos_token_id", 200006)),
                           "pad": int(sid.get("tokenizer.ggml.padding_token_id", 200056))}
    meta["pre_pattern"] = ST.MINIMAX_PATTERN
    meta["pre_pattern_source"] = ("o200k's split with \\p{M} in its letter classes (MiniMax's tokenizer.json, without "
                                  "its NFC): llama.cpp's LLAMA_VOCAB_PRE_TYPE_INKLING, ggml-org/llama.cpp#25731")
    (tok / "tokenizer.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if not (tok / "chat_template.jinja").exists():
        raise SystemExit(f"{a.gguf}: no chat template in the GGUF")
    print(f"{tok}: vocab {meta['vocab_size']}, merges {meta['n_merges']}, pre {meta['pre']}, "
          f"specials {meta['special_ids']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
