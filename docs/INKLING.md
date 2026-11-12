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
the PR's commit 48f7e71, and the six at the card's sampling (temperature 1, top-p 1:
`test-results\2026-11-04-inkling\ink-sampling.py`). maya-ink's runs each start from the same expert profile
(`ink-ab.ps1` there: the engine saves what it learned after every request), its numbers the average of 4 runs (the
first version) and 3 (now):

| | llama.cpp, `--cpu-moe` | llama.cpp, `--moe-cache-mib 14000 -ub 2048` | maya-ink, first version | maya-ink |
|---|---|---|---|---|
| Writing, new prompts | 9.1 tok/s | 9.0 tok/s | 14.5 tok/s | **30.2 tok/s** |
| The same at the card's sampling | - | - | 12.6 tok/s | **31.1 tok/s** |
| Writing after the 31.8K prompt | 9.6 tok/s | 9.7 tok/s | 15.6 tok/s | **26.3 tok/s** |
| Reading the 31.8K prompt | 72 tok/s | 207 tok/s | 733 tok/s | **794 tok/s** |
| Its first token | 441 s | 154 s | 43.5 s | **40.4 s** |
| VRAM | 11.6 GB | 26.8 GB | 29.9 GB | 29.9 GB |

The cache's settings and sizes were chosen by replaying the routing of an ab.py run, so they were checked on prompts
none was tuned on: `test-results\2026-11-04-inkling\ab-heldout.py` is ab.py with six other new prompts (Go, SQL and
Java code, biology, French, German, Japanese), another repeated one (C++) and another long one (21.7K tokens of
Python), every run started from the same profile, which had seen neither set (writing speed, tok/s; ab.py / held-out):

| | New prompts | After the long prompt | The repeated prompt, 3rd time |
|---|---|---|---|
| maya-ink, first version | 14.5 / 14.1 | 15.6 / 16.3 | 16.1 / 19.4 |
| this build, the first version's cache settings (decay 0.998, 4 swaps every 8) | 22.7 / 20.8 | 24.7 / 29.2 | 27.7 / 37.8 |
| this build | **30.2 / 30.2** | 26.3 / 26.3 | **38.7 / 46.6** |

(3 runs each, 4 for the first version on ab.py.) The settings gain as much on the held-out set as on the tuning set:
+45% and +33% on new prompts, +23% and +40% on the repeated one; replaying each set's routing gives the same order
(43%, 57%, 75% of a token's experts in VRAM on the held-out set; 43%, 56%, 72% on ab.py's). Right after a long prompt
they gain nothing: on three long prompts (ab.py's C, the held-out Python, llama.cpp's C++; `ink-deep.py`, 3 runs each)
the new settings wrote 24.2 tok/s and the old ones 25.4, though replaying the same routing gives both the same 71% of
the experts in VRAM there (the prompt's counts fill the cache; few swaps follow) - why they are slower there is open.

Against llama.cpp's own outputs (the same token ids fed to both, greedy, each step's top 5 compared): the chat
template's text and the tokenizer's ids are the same as llama.cpp's on all 4 prompts (27 to 3,054 tokens), the top
token agrees at 87 of 88 positions (the other a near tie: -0.68 against -0.69), and llama.cpp's tokens'
log-probabilities differ by at most 0.34 (69 of 88 by under 0.01), as in the first version (87 of 88, 0.33). Which
experts run on the GPU (int8 activations a 32 values) or the CPU (ggml's a 256) moves the last digits, so a near tie
can go either way; every change that keeps the arithmetic was checked to give the same bits as the build before it.

Where the time went, and what fixed it:

- **The mapping's first touch.** Windows costs ~1 us a page fault even with the file in RAM: the prompt path's copies
  moved 3.4 GB/s until the engine touched every routed expert's page at the start (157 GB in ~11 s, all threads); the
  pages then stay in its working set (about 164 GB: ~22 GB of this machine's RAM stays free while it runs).
- **The driver's staging of pageable memory** moved 4 to 15 GB/s from one run to the next; the engine's own (the CPU
  lane's threads into two 64 MB pinned buffers, the DMA from there) holds 14 GB/s, what the RAM gives three passes over
  the bytes. The link itself does 28.7 GB/s from pinned memory, but Windows will not pin 150 GB of the mapping
  (`cudaHostRegister` there "succeeds" without speeding anything up).
- **The token path** (`INK_STEP_TIMES=1` prints where a token's time went, every 64 tokens). The first version took
  60-75 ms a token: the CPU lane 44-57 ms (125-166 of a token's 240 routed experts on the CPU at 0.35 ms each: its 871
  slots held ~35% of them), the attention and its host sync 8.5 ms, rising to 16.7 by position 256 (attn_decode walked
  each split's 256 keys one after another), ~5 ms of launches around the CPU lane, up to 2.4 ms of the cache's copies,
  and at temperature 1 another 13 ms to sort the vocabulary. Now, on a 256-token answer: the CPU lane ~27 ms (~90
  experts at 0.30 ms, ~50 GB/s, what this RAM gives), the attention and router ~6.5 ms (a layer's graph ~115 us on the
  GPU, then ~40 us of WDDM's submission and the sync's wake-up), the rest ~1.5 ms; on ab.py's new prompts the cache
  finds about two thirds. A `cudaStreamQuery` flush after the launches changed nothing; the graphs saved ~2 ms a token,
  fusing their 22 nodes into 13 about 1%, and a single block doing 4,096 channels' convolutions was slower than the
  kernels it replaced. More speed now needs more of the experts in VRAM.

The hard test (`C:\llm\test-results\2026-11-01-hard\hard-test.py`, 25 tasks) at its best, 2026-11-05: the card's
sampling (temperature 1.0, top-p 1.0), `reasoning_effort` `max`, a 262,144-token budget trimmed to the 128K context
(`fit_max_tokens`; maya-ink has no thinking cap), `--loop-guard` and `--tool-thinking`. It scored 21 of 25 in 279
minutes, 671K tokens at 40 tok/s on average. That's above ab.py's 30, likely because the cache adapts over a long
answer. Every task outside coding came out right, and the repair job passed 10 of 10. Coding was 3 of 7: the Scheme
interpreter (the first of any model's runs there), the ordered multiset and the diff. The regex engine calls its own
helper with an extra argument, and the JSON parser refuses the valid `1e400`. The Sudoku solver took 19.6 s against a
10 s limit, and the SAT solver ran past its 90 s. Both did the same in two more runs, and the engine uses no CPU
between requests. Flash-Next on Strata scores 22 in 51 minutes, Qwen3.8-27B 20 in 130.

### Two cards: an RTX 3090 as the helper card (2026-11-11)

The RTX 5090 (PCIe 5.0 x8) with an RTX 3090 (PCIe 4.0 x8, limited to 320 W) as `--helper-gpu` (branch
`inkling-2gpu`), UD-Q4_K_XL, 128K context. `ab.py` through Maya's server, every run from the same expert profile
(`test-results\2026-11-11-two-gpu\fork-2gpu.ps1`); the one-card column is the same build without the helper. One run
each, and single runs move: two earlier two-card runs with the swaps still on the token's thread wrote 35.5 and 35.4
tok/s on new prompts, 25.4 and 28.2 after the long prompt, and 43.5 and 44.0 on the repeated prompt's third time.

| | The 5090 alone | With the helper card |
|---|---|---|
| Writing, new prompts | 29.6 tok/s | **36.6 tok/s** |
| Writing after the 31.8K prompt | 27.0 tok/s | 28.0 tok/s |
| Reading that prompt | 788 tok/s (first token 40.7 s) | 735 tok/s (45.3 s) |
| A repeated prompt, 1st and 3rd time | 31.4, 37.5 tok/s | 40.3, **47.5 tok/s** |
| First token, short prompts | 1.93 s | 1.50 s |

- **The same answers.** Fed llama.cpp's own tokens (`fork-ref.py ink` there: `ink-ref.py compare` with this build),
  the top token is llama.cpp's at 86 of 88 positions (23/24, 24/24, 23/24, 16/16; one card: 87 of 88). Both misses
  are near ties between the same two tokens (llama.cpp's -0.57 against -0.90, and -0.65 against -0.78), and
  llama.cpp's tokens' log-probabilities differ by at most 0.33, as on one card: the helper's sum joins the CPU lane's
  in another order. The chat checks pass 6 of 6 and `derail.py` finds 0 of 20 derailed (295 s).
- **What it buys.** The helper holds 1,527 experts (23.4 GB) beside the main card's 1,601, of 10,240. On one 600-token
  answer 83.8% of a token's routed experts were then in VRAM instead of 69.8%, and the CPU lane ran in 22.6 of the 40
  MoE layers a token instead of 30.7: 36.0 tok/s against 27.1.
- **Where a token's 27.8 ms goes there:** the CPU lane 17.0 ms (it is bound by the RAM: ~0.3 ms for each 15 MB
  expert), the wait for the helper after it 2.5, and 8.3 ms for the main card's own layers and syncs.
- **Right after a long prompt it gains little** (28.0 against 27.0 through the server; 22.6 against 19.3 on a direct
  400-token run with thinking on). The prompt's counts fill both caches with what the prompt used, few swaps follow
  for some hundred tokens (as on one card, above), and 71% of the experts are in VRAM instead of 49%. The helper's
  refill after a long prompt copies all its slots again: 4.6 s more before the first token.
- **The swaps had to leave the token's thread.** A 15 MB expert from the mapping takes ~0.8 ms to copy, and the tier
  below the main card's experts changes all the time: at 4 swaps every 2 tokens the helper spent 2.3 to 2.6 ms of
  every token copying (16 every 2 tokens: 3.7 ms and no more hits). A thread of its own copies them now (0.04 ms a
  token left), though the CPU lane then shares the RAM with it and takes ~1.3 ms longer: 31.4 against 30.6 tok/s on
  one 300-token answer. Asking more of a swap (`INK_HELPER_BIAS` 0.5 / 2 / 4 / 8: 35.4 / 36.0 / 34.6 / 34.2 tok/s
  on the 600-token answer) is within what single runs move; 2 is the default.
- Both cards stayed cool through these runs: 67 and 50 degrees C at most, 158 and 79 W on average while working.

## The engine

- **The token path** (`Engine::step`): everything but the routed experts on the GPU, int8-activation GEMVs from
  maya-ds4's kernels (src/ds4/ds4_iq.cuh); the router's logits come back to the host, which picks the experts. Those in
  the VRAM cache run on the GPU, the rest on the CPU lane (src/ink/ink_cpu.cpp: maya-ds4's CPU rows read the GGUF's
  mapping in place), at the same time. A MoE layer is one CUDA graph from one host sync to the next (the previous
  layer's end, the attention, the router; `INK_GRAPHS=0`: launches), its small steps fused (`INK_FUSE`, bits 1 the
  norms with their int8 rows, 2 k / v's convolutions, norm and store, 4 q's norm and the bias band inside the
  attention, 8 the output's int8 rows in its combine; 0: none) with the same arithmetic in the same order, so the
  same bits. The attention reads the position from device memory (one grid for every position) and scores a key a
  thread, so the loads are in flight together (it had walked each split's 256 keys one after another, ~1 us a key).
  The CPU lane is one pool run a layer: (expert, band of gate / up rows) jobs, the one finishing an expert's last band
  making its down rows' activation, then bands of down rows summed over the experts there; the workers spin between
  layers (`INK_POOL_SPIN_US`, 3 ms: a sleeping thread takes tens of us to wake on Windows) and start before this
  thread launches the GPU's part (`INK_LANE_EARLY=0`: after).
- **Sampling** (src/ink/ink_main.cpp): at the card's temperature 1 and top-p 1 the token is drawn from the whole
  distribution in one pass, 0.4 ms; sorting all 200K ids had taken 13 ms a token. With top-k, top-p or min-p only as
  many of the most likely ids are sorted as the cut needs. The same distribution, though not the same draw for a seed.
- **The prompt path** (`Engine::prefill`): chunks of up to 8,192 tokens a layer at a time on the GPU. The dense
  matrices are decoded to F16 and multiplied by cuBLAS with F32 sums. The attention is tiles of scores (cuBLAS), a
  softmax that adds the bias band, and the products with V. The router runs in F32. The routed experts go in groups of
  32 (`--stage`): the group's experts some token of the chunk chose, and that are not in the VRAM cache, are copied into
  one of two group buffers (each run of consecutive ones in one copy a tensor, through the engine's pinned staging, on
  a copy stream while the other group is multiplied), then each is decoded, multiplied for its tokens and added to
  theirs times their weights, in id order, then the shared experts: the same sums every run. Prompts under 48 new
  tokens (`INK_MIN_CHUNK`) take the token path.
- **The VRAM expert cache**: the most-used routed experts, in slots sized by their layers' bytes (38 layers' 15.2 MB,
  one layer's 16.3 MB, one's 18.4 MB; each size's count its share of the layers; `INK_CACHE_POOLS=0`: all of the
  largest) in what VRAM is left less 1.5 GB, and in two kinds of lent VRAM: the prompt path's buffers, one 3.9 GB arena,
  between prompts (emptied when a prompt goes through the prompt path, filled at its first token; `INK_LEND=0`), and the
  global layers' K / V rows past the position, 3.6 GB at a conversation's start (each cache's end cut into slots,
  emptied 4,096 rows before the position reaches them; `INK_LEND_KV=0`). The embedding table stays in host memory (a
  token's row is copied up; `INK_EMBD_VRAM=1`: in VRAM), 0.9 GB more for the cache. That is 1,601 slots at 128K
  context against the first version's 871. The scores: each choice by a token or a prompt's token, times 0.99 for each
  token written (`INK_CACHE_DECAY`); every 2 tokens (`INK_CACHE_EVERY`) up to 16 swaps a slot size
  (`INK_CACHE_SWAPS`) of experts that score more than 1.5 times (`INK_CACHE_MARGIN`) + 0.5 the worst one in VRAM;
  refilled after each prompt.
  A swap's copy goes up in 2 MB pieces (`INK_CACHE_STAGE_KB`) while the CPU waits for a layer's attention, when its RAM
  is idle, and the expert is used from the token after its last piece (the same tokens every run; `INK_CACHE_DEFER=0`:
  copied at the swap). `--profile FILE` keeps the scores for the next start. The settings came from replaying the
  routing of an ab.py run offline (`INK_ROUTE_LOG=FILE` writes it; `test-results\2026-11-04-inkling\ec-sim.py`
  replays it, predicting 137.5 experts a token on the CPU where the engine measured 137.7): the first version's
  (decay 0.998, 4 swaps every 8 tokens) left the cache on what the last few hundred tokens had used, where what the
  last few dozen use predicts the next ones better.
- **The helper card** (`--helper-gpu N`, `--helper-gb G`; maya-nem's, branch `nemotron-2gpu`): a second card that
  holds routed experts only, in a cache of its own (slots by size as the main cache's; its free VRAM less
  `INK_HELPER_RESERVE_MIB`, 1024), filled with the best-scored experts the main card does not hold. A token's chosen
  experts are then split three ways: the main card's, the helper's and the CPU lane's. The helper gets the layer's
  input from the host (it is there after the layer's sync), runs the same gate / up, SwiGLU and down launches on its
  own stream while the main card and the CPU lane do theirs, and returns its sum to the host, which adds it to the
  CPU lane's before the main card's next graph reads both. No state of the model lives there, so the prompt path,
  `reset()` and the graphs are as before, and without `--helper-gpu` the engine runs as it did.
  - The main cache decides first and leaves the helper's experts alone: its swaps take only experts on neither card
    (`INK_HELPER_STEAL=1`: the helper's too).
  - The helper follows: every 2 tokens (`INK_HELPER_EVERY`) up to 4 swaps (`INK_HELPER_SWAPS`), an expert the main
    card took dropped, and one from outside both taken in place of its worst when it scores more than the main
    cache's margin times that one's + 2 (`INK_HELPER_MARGIN`, `INK_HELPER_BIAS`). A thread of its own copies them on a
    second stream while the next tokens are written, and an expert is used from the first update after its copy
    (`INK_HELPER_BG=0`: copied at the swap, on the token's thread). At the start and after a prompt the fill is
    copied at once: after a prompt up to an eighth of its tokens (64 .. 4,096, `INK_HELPER_FILL`).
  - Maya's server adds `--layer-split auto` when a config's `"gpu"` names two cards; this engine reads that as
    `--helper-gpu 1` (the second visible card), so `"gpu": [0, 1]` in the run config is all it takes
    (`INK_HELPER_GPU=-1` in its `env`: off).
  - The kernels launched there (ds4_iq's GEMVs, ink's q8 / SwiGLU / row sums) keep no device state, so they run on
    whichever device is current; the build needs both cards' architectures (`-DCMAKE_CUDA_ARCHITECTURES=86;120`).

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
