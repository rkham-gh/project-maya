# Inkling-Small on Maya (maya-ink)

Thinking Machines' Inkling-Small (released 2026-07-30, Apache 2.0) is a 276B mixture of experts with 12B active per
token. `maya-ink` (src/ink) runs it from Unsloth's GGUF on one NVIDIA GPU and the CPU, behind Maya's own server
(serve/server.py), the way maya-ds4 runs DeepSeek. llama.cpp can run it only with the open draft PR
ggml-org/llama.cpp#25731. That PR's graph (src/models/inkling.cpp) is the reference this engine follows, and its build is
what the measurements below compare against.

## The model

| | |
|---|---|
| Layers | 42. The first 2 have a dense SwiGLU FFN (16,384 wide), the other 40 a MoE |
| Width | 4,096. 32 query heads and 8 KV heads of 128 |
| Attention | No RoPE. 35 sliding layers (window 512), 7 global ones (every sixth), a learned relative-position bias band (d_rel 16; 1,024 positions on the global layers, 512 on the sliding ones), score = q.k / 128 + bias |
| Short convolutions | 4 a layer (k, v, the attention's output, the FFN's output): x + a causal depthwise conv of width 4 |
| MoE | 256 routed experts (2,048 wide), top 6 of sigmoid(logit) + bias, weights softmax(logsigmoid) over the 6 and the 2 shared experts' logits, times 8 |
| Long context | q and the bias times tau = 1 + 0.1 ln(max((pos + 1) / 128,000, 1)) on the global layers; 1M positions |
| Vocabulary | 201,024 (o200k's split with \p{M} in its letter classes; 200,058 real ids, the rest masked) |

UD-Q4_K_XL is 163 GB: every dense matrix (attention, the dense FFNs, the shared experts, the output head, the
embedding) Q8_0, 6.5 GB; the routed experts Q4_K / Q5_K / Q6_K, 150 GB.

## Measured

An RTX 5090 at PCIe 5.0 x8, a Ryzen 9 9950X, 192 GB of DDR5-3600, Windows 11, CUDA 13.4, UD-Q4_K_XL, 128K context.
`ab.py` (temperature 0, thinking off; six new prompts of 400 tokens, then a 31.8K-token prompt) against llama.cpp at
the PR's commit 48f7e71:

| | llama.cpp, `--cpu-moe` | llama.cpp, `--moe-cache-mib 14000 -ub 2048` | maya-ink |
|---|---|---|---|
| Writing, new prompts | 9.1 tok/s | 9.0 tok/s | **15.6 tok/s** |
| Writing after the 31.8K prompt | 9.6 tok/s | 9.7 tok/s | **16.1 tok/s** |
| Reading the 31.8K prompt | 72 tok/s | 207 tok/s | **745 tok/s** |
| Its first token | 441 s | 154 s | **43 s** |
| VRAM | 11.6 GB | 26.8 GB | 29.9 GB |

Against llama.cpp's own outputs (the same token ids fed to both, greedy, each step's top 5 compared): the chat
template's text and the tokenizer's ids are the same as llama.cpp's on all 4 prompts (27 to 3,054 tokens), the top
token agrees at 87 of 88 positions (the one other was a near tie), and llama.cpp's tokens' log-probabilities differ
by at most 0.33 (most by under 0.01) - with the expert cache on and off, through the token path and the prompt path.

Where the time went, and what fixed it:

- **The mapping's first touch.** Windows costs ~1 us a page fault even with the file in RAM: the prompt path's copies
  moved 3.4 GB/s until the engine touched every routed expert's page at the start (157 GB in ~11 s, all threads); the
  pages then stay in its working set (about 164 GB: ~22 GB of this machine's RAM stays free while it runs).
- **The driver's staging of pageable memory** moved 4 to 15 GB/s from one run to the next; the engine's own (the CPU
  lane's threads into two 64 MB pinned buffers, the DMA from there) holds 14 GB/s, what the RAM gives three passes over
  the bytes. The link itself does 28.7 GB/s from pinned memory, but Windows will not pin 150 GB of the mapping
  (`cudaHostRegister` there "succeeds" without speeding anything up).
- **The token path** spends ~70% of its time on the CPU lane (the experts not in VRAM, ~38-40 GB/s from RAM), the rest
  in ~40 kernel launches and a host sync a layer. At 128K context the cache holds 871 of the 10,240 experts; with 1,063
  (8K context) it found 48-72% of a token's experts in VRAM on the test prompts (15-26 tok/s). CUDA graphs and lending
  the prompt path's 3.9 GB to the cache are the next steps.

## The engine

- **The token path** (`Engine::step`): everything but the routed experts on the GPU, int8-activation GEMVs from
  maya-ds4's kernels (src/ds4/ds4_iq.cuh); the router's logits come back to the host, which picks the experts. Those in
  the VRAM cache run on the GPU, the rest on the CPU lane (src/ink/ink_cpu.cpp: maya-ds4's CPU rows read the GGUF's
  mapping in place), at the same time.
- **The prompt path** (`Engine::prefill`): chunks of up to 8,192 tokens a layer at a time on the GPU. The dense
  matrices are decoded to F16 and multiplied by cuBLAS with F32 sums. The attention is tiles of scores (cuBLAS), a
  softmax that adds the bias band, and the products with V. The router runs in F32. The routed experts go in groups of
  32 (`--stage`): the group's experts some token of the chunk chose, and that are not in the VRAM cache, are copied into
  one of two group buffers (each run of consecutive ones in one copy a tensor, through the engine's pinned staging, on
  a copy stream while the other group is multiplied), then each is decoded, multiplied for its tokens and added to
  theirs times their weights, in id order, then the shared experts: the same sums every run. Prompts under 48 new
  tokens (`INK_MIN_CHUNK`) take the token path.
- **The VRAM expert cache**: what VRAM is left (less 1.5 GB) holds the most-used routed experts. Every choice by a
  token or a prompt's token counts, the counts decay as tokens are written, and the cache is refilled after each prompt
  and adapted every 8 tokens (4 swaps at most). `--profile FILE` keeps the counts for the next start.

## The server

`serve/frontend.py` knows the template by `<|content_thinking|>` (`ChatTemplate.inkling`):

- Thinking levels: the client's word when Inkling has it (none, low, medium, high, xhigh, max; `minimal` counts as
  off), thinking off is `none`. The template's default is high (0.9); TML's evaluations ran at 0.99 (`max`).
- Tool results are written as their tool's own messages (`ink_tool_names`), as the template wants.
- `InklingRewriter` turns the typed blocks the model writes into what `OutputParser` reads: a thinking block becomes
  reasoning, a text block the answer, and `NAME<|content_invoke_tool_json|>{"name": .., "args": {..}}<|end_message|>`
  a tool call. `serve/test_inkling_frontend.py` checks the template and the parser.
- `serve/server.py`: `<|content_model_end_sampling|>` ends a turn; there is no thinking budget (no token ends the
  thinking alone); a forced tool call is prefilled with the tool's name and its JSON's opening.

`tools/ink/make_pack.py` writes the tokenizer folder the server reads (with the template the GGUF carries) - the
engine reads the weights from the GGUF itself:

```
python tools/ink/make_pack.py --gguf <...-00001-of-00005.gguf> --out <pack dir>
```

A run config (`--engine maya --config ...`) names `build-ink/maya-ink.exe` and its arguments:

```json
{
  "exe": "<fork>\\build-ink\\maya-ink.exe",
  "args": ["--gguf", "<...-00001-of-00005.gguf>", "--max-context", "131072", "--profile", "<pack dir>\\expert-profile.bin"],
  "tokenizer": "<pack dir>\\tokenizer",
  "model_name": "inkling-small",
  "gpu": [0],
  "sampling": {"temperature": 1.0, "top_p": 1.0},
  "reasoning_effort": "high"
}
```

The engine's own test mode compares against another engine token by token:

```
maya-ink --gguf <...> --ids @prompt.txt --gen 24            greedy, each step's top 5 as id:log-probability
maya-ink --gguf <...> --ids @prompt.txt --follow @their.txt  their tokens fed, each one's log-probability here
```
