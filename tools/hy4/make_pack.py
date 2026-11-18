#!/usr/bin/env python3
"""Hy4 Flash's tokenizer pack for maya-hy4: the tokenizer and the chat template its GGUF carries (tools/strata_tokenizer.py's
extract), with its end-of-turn token named the way the server reads it.

    python tools/hy4/make_pack.py --gguf <...-00001-of-0000N.gguf> --out <pack dir>      (writes <pack dir>/tokenizer)

The server (serve/server.py) reads the pack's tokenizer/ (its chat template included) and starts build-hy4/maya-hy4 with
the GGUF itself: maya-hy4 reads the weights straight from the shards, so the pack holds no weights.
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
    ap.add_argument("--gguf", required=True, help="the first shard of Hy4 Flash's GGUF")
    ap.add_argument("--out", required=True, help="the pack directory (tokenizer/ goes inside)")
    a = ap.parse_args()
    tok = pathlib.Path(a.out) / "tokenizer"
    cfg = ST.extract(a.gguf, a.out)   # (it writes <out>/tokenizer/)
    if cfg["pre"] != "hy_v4":
        raise SystemExit(f"{a.gguf}: pre-tokenizer {cfg['pre']!r}, not Hy4's 'hy_v4'")
    meta = json.loads((tok / "tokenizer.json").read_text(encoding="utf-8"))
    sid = meta.get("special_ids") or {}
    # (the server ends a request at the "eos" a pack names; the turn's own end, <｜hy_end:...｜>, is a stop id too)
    meta["special_ids"] = {"bos": int(sid.get("tokenizer.ggml.bos_token_id", 120000)),
                           "eos": int(sid.get("tokenizer.ggml.eos_token_id", 120025)),
                           "pad": int(sid.get("tokenizer.ggml.padding_token_id", 120002))}
    meta["pre_pattern"] = list(ST.DEEPSEEK3_PATTERNS)
    meta["pre_pattern_source"] = ("llama.cpp's LLAMA_VOCAB_PRE_TYPE_HY_V4 (ggml-org/llama.cpp#32517, src/llama-vocab.cpp): "
                                  "three Isolated splits, DeepSeek-V3's; no normalizer")
    (tok / "tokenizer.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if not (tok / "chat_template.jinja").exists():
        raise SystemExit(f"{a.gguf}: no chat template in the GGUF")
    print(f"{tok}: vocab {meta['vocab_size']}, merges {meta['n_merges']}, pre {meta['pre']}, "
          f"specials {meta['special_ids']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
