# Nemotron 3.5 Super on Maya (maya-nem)

NVIDIA's Nemotron 3.5 Super VL (released 2026-10-15, OpenMDW-1.1) is a 120B hybrid Mamba-2 / attention mixture of
experts with 12B active per token. `maya-nem` (src/nem) runs its text side from Unsloth's GGUF on one NVIDIA GPU and
the CPU, behind Maya's own server (serve/server.py), the way maya-ink runs Inkling-Small (branch `inkling`, whose
machinery this engine is built from). Mainline llama.cpp runs the model too (arch `nemotron_h_moe`, since
ggml-org/llama.cpp#30520); its graph (src/models/nemotron-h.cpp and mamba-base.cpp's `build_mamba2_layer`) is the
reference this engine follows, and its build is what the measurements compare against.

## The model

| | |
|---|---|
| Layers | 88: 40 Mamba-2, 40 MoE, 8 attention (7, 16, 25, 36, 47, 58, 69, 78), each `x += block(rmsnorm(x))`; plus one NextN (MTP) block, not used here |
| Width | 4,096 |
| Mamba-2 | 128 heads of 64, state 128, 8 groups (B / C), a causal conv of 4 over x \| B \| C (10,240 channels) with a bias and SiLU, dt = softplus(dt + bias), the gate silu(z), a grouped RMS norm (groups of 1,024), `ssm_in` 4,096 -> 18,560, `ssm_out` 8,192 -> 4,096 |
| Attention | 32 query heads, 2 KV heads of 128, no RoPE, scale 1/sqrt(128): 8 KiB of K / V a token |
| MoE | 512 routed experts in a 1,024-wide latent space (`ffn_latent_down` 4,096 -> 1,024, each expert up 1,024 -> 2,688, relu^2, down 2,688 -> 1,024, `ffn_latent_up` 1,024 -> 4,096), top 22 of sigmoid(router) + bias, weights the chosen sigmoids normalized x 5; one shared expert (4,096 -> 5,376 -> 4,096, relu^2) |
| Vocabulary | 131,072, the `tekken` pre-tokenizer (llama.cpp's LLAMA_VOCAB_PRE_TYPE_TEKKEN: ignore_merges), no BOS added |
| Template | ChatML; thinking on by default (`<think>\n` opens the answer), `enable_thinking`, `medium_effort` ("{reasoning effort: efficient}" after the last user message); tools in Qwen3-Coder's XML |

UD-Q4_K_XL is 86.9 GB: the routed experts 75.8 GB (up rows Q5_K / Q4_K / Q6_K, down rows of 2,688 values Q5_0 /
IQ4_NL, about 5.4 bits a weight), the Mamba-2 and attention layers Q8_0 / Q6_K, the shared experts Q6_K / Q8_0, the
output head and the embedding Q8_0, the NextN block Q8_0 (3.1 GB).

## The engine

- **The token path** (`Engine::step`): everything but the routed experts on the GPU, int8-activation GEMVs from
  maya-ds4's kernels (src/ds4/ds4_iq.cuh, Q5_0 added for the experts' down rows). The Mamba-2 mixer's conv and
  selective scan are this engine's kernels (src/nem/nem_ext.cu: a warp two rows of a head, its state in registers
  over the tokens); the attention is maya-ink's split-key decode kernel. A MoE layer's router logits and latent go to
  the host, which routes; the chosen experts in the VRAM cache run on the GPU (up, relu^2, down in the latent space),
  the rest on the CPU lane (src/nem/nem_cpu.cpp, maya-ds4's CPU rows on the GGUF's mapping), at the same time, with
  the shared expert on the GPU. Between the 40 syncs the GPU's work is 41 CUDA graphs (`NEM_GRAPHS=0`: launched one
  by one).
- **The prompt path** (`Engine::prefill`): chunks of up to 8,192 tokens a layer at a time on the GPU. The dense
  matrices are decoded to F16 and multiplied by cuBLAS with F32 sums; a Mamba-2 layer runs in sub-chunks of 2,048
  tokens (the projection's rows, the conv and the scan carried over them); the attention is tiles of causal scores
  (cuBLAS), a softmax and the products with V; the router runs in F32. The routed experts go in groups of 64
  (`--stage`): the group's experts some token chose that aren't in the VRAM cache are copied into one of two group
  buffers through pinned staging, then each is decoded and multiplied for its tokens, in id order. Prompts under 32
  new tokens (`NEM_MIN_CHUNK`) take the token path.
- **The VRAM expert cache**: maya-ink's - slots sized by their layers' experts, the prompt path's buffers and the
  attention layers' K / V rows past the position lent to it between prompts, scores that count every choice and decay
  0.99 a token, up to 16 swaps every 2 tokens (margin 1.5; `NEM_CACHE_*`), the swaps' copies staged while the CPU
  waits for the GPU. `--profile FILE` keeps the scores for the next start.
  A prompt chunk adds each expert's count of its tokens as it is: thousands after a long prompt, where a written
  token adds 1 and an expert every token chose settles at 99. The refill after the prompt so holds what the prompt
  used most, and no swap clears the bar (1.5 times the worst one in VRAM + 0.5) until those scores have decayed, some
  hundred tokens on. `NEM_CACHE_PROMPT=1` (off by default, not run yet; maya-ink's) puts a chunk on the written
  tokens' scale: each expert gets its share of the chunk's tokens times what as many written tokens leave to share
  (at most 99), so the refill takes the same experts and the swaps start with the first tokens; `2` also decays what
  was there before the chunk, as its tokens written one by one would.
- **The helper card** (`--helper-gpu N`, branch `nemotron-2gpu`): a second card that holds routed experts only, in a
  cache of its own (slots by class, as the main cache's; its free VRAM less `NEM_HELPER_RESERVE_MIB`, 1024). It is
  filled with the best-scored experts the main card does not hold. A token's chosen experts are then split three
  ways: the main card's, the helper's, and the CPU lane's. The helper gets the latent from the host (it is there
  after the MoE layer's sync), runs the same up / relu^2 / down launches on its own stream while the main card and
  the CPU lane do theirs, and returns its sum to the host, which adds it to the CPU lane's before the main card's
  next segment reads both. No state of the model lives on the helper, so the prompt path, `reset()` and the CUDA
  graphs are as before, and without `--helper-gpu` the engine runs exactly as it did.
  - The main cache decides first and leaves the helper's experts alone: its swaps take only experts that are on
    neither card (`NEM_HELPER_STEAL=1` lets it take the helper's too: 41 tok/s instead of 55 on one 800-token run,
    because every such swap empties a helper slot and sends the main card's evicted expert to the CPU).
  - The helper follows: up to 16 swaps every 2 tokens (`NEM_HELPER_SWAPS`, `NEM_HELPER_EVERY`), an expert the main
    card took dropped, a better-scored one from outside both taken in place of its worst (the main cache's margin);
    after a prompt, up to an eighth of the prompt's tokens (64 .. 4,096, `NEM_HELPER_FILL`).
  - The fills at the start and after a prompt go through pinned staging of the helper's own, as the main cache's do
    (`hc_copy_many`, maya-ink's: the CPU lane's threads into two 64 MB pinned buffers in turn, one DMA an expert
    from there; `NEM_HELPER_STAGE=0`: the driver's staging of the mapping, as before; another value: a buffer's MB).
    Not run on a GPU yet. `NEM_HELPER_VERIFY=1` reads every slot a fill wrote back and compares it with the
    mapping's bytes.
  - `NEM_HELPER_BG=1` (off by default, not run yet): maya-ink's and maya-ling's swaps' thread. The swaps between
    tokens are copied by a thread of their own on a second stream while the next tokens are written, and an expert
    is used from the first update after its copy is done: the copies (the 0.4 ms of a token below, 0.19 ms each)
    leave the token's thread, so more of them a token cost it nothing (new prompts write 48 tok/s where long answers
    reach 55, the helper taking new experts in at 8 a token). `NEM_HELPER_MARGIN` / `NEM_HELPER_BIAS` set a swap's
    bar (the main cache's margin and 0.5, as before); maya-ling's helper runs 32 swaps every token this way.
  - Maya's server adds `--layer-split auto` when a config's `"gpu"` names two cards; this engine reads that as
    `--helper-gpu 1` (the second visible card), so `"gpu": [0, 1]` in the run config is all it takes.
  - The kernels it launches there (ds4_iq's GEMVs, nem's q8 / relu^2 / row sums) keep no device state, so they run
    on whichever device is current; the build needs both cards' architectures (`-DCMAKE_CUDA_ARCHITECTURES=86;120`).
- At start the engine touches every routed expert's pages (75.8 GB, ~5 s from a warm file cache), as maya-ink does:
  on Windows a page's first touch costs a fault even with the file cached.

## Measured

An RTX 5090 at PCIe 5.0 x8, a Ryzen 9 9950X, 192 GB of DDR5-3600, Windows 11, CUDA 13.4, UD-Q4_K_XL.

Against mainline llama.cpp (b12017, `--cpu-moe`), the same token ids fed to both, greedy, each step's top 5 compared
(`test-results\2026-11-04-nemotron\nem-ref.py`): the chat template's text and the tokenizer's ids identical on 4
prompts (28 to 3,621 tokens); the top token the same at 88 of 88 positions through the prompt path and 47 of 48
through the token path alone (the other a near tie, 0.015 apart in llama.cpp); llama.cpp's token's log-probability
within 0.067. The CUDA graphs give the same bits.

Through the server (`nem-chat-test.py`): an answer with its thinking apart, thinking off, a tool call, the answer from
the tool's result, streaming and JSON output - 6 of 6. `derail.py`: 0 of 20 rounds derailed.

Speed, on a quiet machine, with `ab.py` (temperature 0, thinking off; six new prompts of 400 tokens, then a 34.2K-token
prompt, then one prompt three times), at 256K context; maya-nem's runs each started from the same expert profile
(`nem-ab.ps1`: the scores after the chat checks from an empty start), 4 runs of the defaults (8K chunks) and 3 of
`--chunk 16384`, the run config's:

| | llama.cpp, `-ngl 99 --cpu-moe` | llama.cpp's own fit, `-ub 2048 -b 2048` | maya-nem, 8K chunks | maya-nem, 16K chunks |
|---|---|---|---|---|
| Writing, new prompts | 11.2 tok/s | 13.0 tok/s | **26.6 tok/s** | 26.3 tok/s |
| Writing after the 34.2K prompt | 11.4 tok/s | 13.5 tok/s | 21.6 tok/s | **21.7 tok/s** |
| Reading the 34.2K prompt | 107 tok/s | 474 tok/s | 1,130 tok/s | **1,607 tok/s** |
| Its first token | 319 s | 72 s | 30 s | **21 s** |
| The repeated prompt, 1st -> 3rd | 11.4 -> 11.4 | 13.5 -> 13.5 | 24.0 -> 29.6 tok/s | 23.9 -> 29.3 tok/s |
| VRAM | 12.2 GB | 30.3 GB | 29.9 GB | 29.9 GB |

- The cache settings: maya-ink's tuned ones (0.99 / 16 swaps / every 2 / margin 1.5) against its earlier ones (0.998 / 4
  / 8 / 1.25), 2 runs each, alternated: new prompts 26.6 against 22.1 tok/s, the repeated prompt's third answer 29.6
  against 22.9, after the long prompt the same (21.5 against 21.5).
- 16K chunks read the long prompt 42% faster (each expert crosses PCIe once for twice the tokens) and write about 1%
  slower on new prompts (26.2-26.4 against 26.5-26.6 tok/s): the prompt path's buffers are lent to the expert cache
  between prompts, 1,059 slots of the 6,119 at 16K against 753 at 8K.
- With llama.cpp's fit, `-ngl` must stay unset: given `-ngl 99` without `--cpu-moe`, it put the whole model "on" the
  GPU, 32.0 GB of VRAM with the rest spilled to shared memory, and wrote 3.0 tok/s.
- About 80% of a token is the CPU lane's share of the experts, read at the RAM's speed (~40 GB/s); the CUDA graphs
  gained ~5%. More writing speed needs more of the experts in VRAM.

Unsloth's UD-Q3_K_XL (65.8 GB; 2026-11-05) is mostly 4-bit IQ formats despite the name. Its routed experts are IQ4_NL
28.5 GB, IQ4_XS 15.7, IQ3_XXS 8.1, IQ2_XS 1.6 and Q2_0 1.6: 55.6 GB against Q4's 75.8. Q2_0 is ggml's type 42, blocks of
64 with 2-bit codes, value (q - 1) d, the 2-bit fallback for the down rows of 2688. src/ds4/ds4_iq now has it in the
GEMVs, FP16 rows and decode_values; the CPU lane reads it through ggml-cpu's dot, which has no x86 SIMD version (the
AVX2 rows below replace it). Measured the same way, 3 runs:

| UD-Q3_K_XL, 16K chunks | maya-nem |
|---|---|
| Writing, new prompts | **37.7 tok/s** (37.7 / 37.6 / 37.7), +43% |
| Writing after the 34.2K prompt | **30.6 tok/s** (30.9 / 30.7 / 30.3), +41% |
| Reading the 34.2K prompt | **1,981 tok/s**, +23% |
| Its first token | 17.3 s |
| The repeated prompt, 1st -> 3rd | 35.0 -> 41.1 tok/s |
| VRAM | 29.9 GB: 6,555 expert slots + 1,283 in the prompt path's buffers + 688 in K / V rows (8,526 against Q4's 6,119) |

- Against llama.cpp on the same file (`nem-ref.py --ref q3`): the top token is the same at 86 of 88 positions. The
  other two are near ties: at each, two tokens were around -0.66 to -0.81, and the order flipped. llama.cpp's token's
  log-probability is within 0.141.
- Against Q4 (`test-results\2026-11-04-nemotron\q3-agree.py`): Q3 was fed Q4's greedy continuations of 3 prompts. Its
  top token was Q4's at 949 of 976 positions (97.2%). Q4's token's log-probability was 0.026-0.030 apart on average,
  and 1.07 at worst.
- The chat checks passed 6 of 6, and `derail.py` found 0 of 20 rounds derailed. The text was the same on all 3 runs.
- Q4 is unchanged by the Q2_0 build. With the earlier cache settings (`NEM_CACHE_DECAY=0.998`, `_EVERY=8`,
  `_SWAPS=4`, `_MARGIN=1.25`), it gives the earlier build's log-probabilities bit for bit. With the current settings
  it differs slightly at the token path's positions, as the cache settings change which experts run on the GPU.

The hard test (`test-results\2026-11-01-hard\hard-test.py`, 25 tasks) on UD-Q3_K_XL at its best, 2026-11-05: the
card's sampling (temperature 1.0, top-p 0.95), a 262,144-token budget trimmed to the context (`fit_max_tokens`), the
thinking cap off (`thinking_budget` 0), `--loop-guard` and `--tool-thinking`. It scored 17 of 25 in 195 minutes, 472K
tokens at 40 tok/s on average. Every task outside coding came out right, though trace-3's line came without the code
block the prompt asks for, so it scores as wrong. The repair job passed 10 of 10. None of the 7 coding tasks passed:
each answer stopped by itself after 9K-48K tokens with a bug in its own code. Flash-Next on Strata scores 22 in 51
minutes, Qwen3.8-27B 20 in 130.

The AVX2 Q2_0 rows (2026-11-05), taken from a Codex session's `codex/nemotron-mtp`: `src/ds4/ds4_q2_0_avx2.hpp` is an
AVX2 Q2_0 x Q8_0 dot (adapted from llama.cpp's x86 kernels, MIT) that the CPU lane uses for the Q2_0 down rows when
`DS4_Q2_0_AVX2=1` and the CPU has AVX2. Otherwise nothing changes. The Q3 pack's run config sets it. It was measured
in Codex's build, which was the same code with the variable set, against the build before it on Q3, the same way as
above:

| UD-Q3_K_XL, 16K chunks | Before | AVX2 rows |
|---|---|---|
| Writing, new prompts | 37.6 / 37.7 tok/s | **41.2 / 41.4 / 41.3 tok/s** |
| Writing after the 34.2K prompt | 30.3 / 31.9 | **34.5 / 34.4 / 33.5** |
| Reading the 34.2K prompt | ~1,995 tok/s | ~1,973 tok/s |
| Thinking on, the card's sampling (`spec-sweep.py`) | 38.8 / 38.7 | **41.7 / 42.4** |

- The top token was the earlier build's at 88 of 88 positions and llama.cpp's at 86 of 88. The chat checks passed 6
  of 6, and `derail.py` found 0 of 20 rounds derailed. This build wasn't measured again after the change.
- `tests/ds4_q2_0_rows_test.cpp` checks the rows against a scalar reference (ctest `ds4_q2_0_avx2`). With
  `DS4_Q2_0_AVX2=0` the earlier scalar path misses its tolerance in 15 of 3,960 checks. That path is unchanged, and
  this hasn't been looked into.
- The rest of that branch is left out: decoding with the model's MTP head (`--mtp N`). It was exact in every check,
  but 22-31% slower: 28.4 tok/s with 4 drafts, 32.0 with 2 and 32.3 with 1, against 41.3, at 95% accepted with 1. Each
  drafted token still reads its own experts from RAM, the head takes ~4 GB of VRAM from the expert cache, and the
  drafts are checked one token at a time.

### Two cards: an RTX 3090 as the helper card (2026-11-11)

The RTX 5090 (PCIe 5.0 x8) with an RTX 3090 (PCIe 4.0 x8, limited to 320 W) as `--helper-gpu`, UD-Q3_K_XL, 256K
context, the AVX2 Q2_0 rows on. `ab.py` through Maya's server, every run from the same expert profile
(`test-results\2026-11-11-two-gpu\nem-2gpu.ps1`); the one-card column is the same build without the helper.

| | The 5090 alone | With the helper card | llama.cpp, both cards |
|---|---|---|---|
| Writing, new prompts | 41.0 tok/s (39.1 on the serving build the same day) | **48.2 tok/s** | 23.0 tok/s |
| Writing after a 34.2K-token prompt | 33.9 tok/s | **43.8 tok/s** | 25.7 tok/s |
| Reading that prompt | 1,925 tok/s | 1,829 tok/s | 1,079 tok/s |
| A repeated prompt, 1st and 3rd time | 36.8, 44.9 tok/s | 53.7, 50.7 tok/s | 23.9, 23.7 tok/s |
| First token, short prompts | 1.91 s | 2.24 s | 0.52 s |

- **The same answers.** Fed llama.cpp's own tokens for this quant (`nem-ref.py compare ... --ref q3`), the helper
  build's top token was llama.cpp's at 87 of 88 positions (24/24, 23/24, 24/24, 16/16), log-probabilities within
  0.11. The chat checks pass 6 of 6 and `derail.py` finds 0 of 20 derailed (325 s). Greedy text parts ways with the
  one-card run after a few dozen tokens, because the helper's sum joins the CPU lane's in another order.
- **What it buys.** The helper holds 8,620 experts beside the main card's 8,569 (of 20,480). On one 800-token
  answer 94.7% of the routed experts were then in VRAM instead of 74%, and the CPU lane ran in 23.5 of the 40 MoE
  layers a token instead of 38.3: 55.3 tok/s against 39.9.
- **Where a token's 18.1 ms goes there:** the CPU lane 6.7 ms, the wait for the helper after it 1.8, the helper's
  swaps 0.4, and 9.2 ms for the main card's own layers and its 40 syncs, which no second card shortens. So this
  design tops out near 100 tok/s; the next gain is in the main card's token path, not in more VRAM.
- **New prompts gain less than long answers** (48 against 55 tok/s): each new topic needs other experts, and the
  helper takes them in at 8 a token. Faster swaps cost their copies (0.19 ms each over this link).
- The first token of a short turn comes 0.3 s later (the helper's refill after the prompt).

#### The helper's clocks: an opt-in boost (2026-11-13)

A helper's work comes in bursts of tens of microseconds a layer, so its driver keeps an RTX 3090 helper in P3 while
tokens are written: 495-855 MHz, its memory at 5,001 of 9,751 MHz. `NEM_HELPER_KEEPALIVE=1` keeps a one-warp kernel
asleep for `NEM_HELPER_KEEPALIVE_MS` (50) on a lowest-priority stream of the helper, launched again each token once
the last has ended, so the card stays in P2 (2,085 / 9,501 MHz) while writing and idles again within 50 ms of the last
token. `ab.py` on both cards (UD-Q3_K_XL, `NEM_SNAPSHOT=1 NEM_HELPER_BG=1`), the same build off / on, 2 interleaved
runs each: new prompts 56.5 -> 62.5 tok/s, a repeated prompt's 3rd time 58.0 -> 67.0, after the 34K prompt 57.5 ->
57.5; prefill unchanged; the greedy text identical; chat checks 6/6 and `derail.py` 0/20 with it on. It costs the 3090
106 -> 196 W while writing (both cards 361 -> 467 W, 17% more energy a token), so it's off by default.

## The server

- `serve/frontend.py` knows the template by its `medium_effort`: the server's `low` / `medium` (and `minimal`) turn it
  on, the other levels keep the full default; `"chat_template_kwargs": {"enable_thinking": false}` turns thinking off.
- The server's thinking budget (32,768 tokens unless the run config sets `thinking_budget`) applies: `_sliced` closes
  the reasoning at the first sentence or line end past it. maya-nem ignores the `think_budget` / `think_end` keys of
  `GEN`, the engine's own close for reasoning with no such end.
- A prompt is read on from the engine's state only when it continues the tokens so far (the Mamba-2 state can't be
  rewound). A tool's result does (`test-results\2026-11-04-nemotron\nem-turns.py`: 473 of 511 tokens read on); a new
  user message doesn't, because the template drops the earlier thinking (`truncate_history_thinking`), so the
  conversation is read again from the start. `NEM_SNAPSHOT=1` (off by default, not run on a GPU yet) keeps the state
  from just before each prompt's end, where the next turn's rendering first differs (`Engine::snapshot`: the Mamba-2
  layers' convolution inputs and scan states, 173 MB of VRAM; the attention layers' K / V need nothing, a row is only
  ever written at its position). The prompt ends with `<|im_start|>assistant\n<think>\n` and the next prompt writes
  that turn as `...<think></think>` + the answer, so the snapshot is taken 4 tokens before the end
  (`NEM_SNAPSHOT_TAIL`) and those last tokens are read one by one (the token path: their logits come from its int8
  rows, not the prompt path's F16). A new user message then reads on from there: those tokens, the answer without its
  thinking and the message, instead of the whole conversation; the same prompt sent again reads only the tail.
  `maya-nem --gguf <...> --ids @prompt.txt --gen 64 --restore-check` checks it: the state goes back to the prompt's
  end, the same tokens are fed again, and each step's logits are compared with the first pass's (the same bits with
  `NEM_CACHE_SWAPS=0` and no helper card).
- `QUIT` is the last line the engine reads: the thread that reads stdin went back into `getline`, holding the lock
  `exit()` takes to flush the streams, and on Linux the engine then never ended while its stdin stayed open (the
  server waits `MAYA_ENGINE_QUIT_S`, then kills it).
- A prompt chunk's embedding rows go up in staging-buffer pieces (maya-ling's): a 16K chunk's are 71 MB, more than a
  64 MB buffer, and went up a row a call (16,384 copies from pageable memory a full chunk).
- `tools/strata_tokenizer.py` has the `tekken` pre-tokenizer; `tools/nem/make_pack.py` writes the tokenizer folder
  the server reads (with the template the GGUF carries):

```
python tools/nem/make_pack.py --gguf <...-00001-of-00003.gguf> --out <pack dir>
```

A run config (`--engine maya --config ...`):

```json
{
  "exe": "<worktree>\\build-nem\\maya-nem.exe",
  "args": ["--gguf", "<...-00001-of-00003.gguf>", "--max-context", "262144", "--chunk", "16384", "--profile", "<pack dir>\\expert-profile.bin"],
  "tokenizer": "<pack dir>\\tokenizer",
  "model_name": "nemotron-3.5-super",
  "gpu": [0],
  "sampling": {"temperature": 1.0, "top_p": 0.95}
}
```

Not run on a GPU yet (2026-11-12): written in a session without one. All of it builds with CUDA 13.4's nvcc
(Linux), and the host side was run against a stand-in for the CUDA runtime (device memory as host memory, a stream's
copies made only when it is synchronized, kernels computing nothing) on a small synthetic GGUF, with and without a
second device: the slots' bytes after a fill, the swaps' thread at 32 swaps a token, the serving loop's reuse over
seven turns, the exit. That says nothing about the model's numbers or about speed. To be measured as the rest was:

| | Switch | Default | The check |
|---|---|---|---|
| The helper's fills through pinned staging | `NEM_HELPER_STAGE` | on | `NEM_HELPER_VERIFY=1` once; then the first token of a short turn with the helper (2.24 s against 1.91 s without) |
| The state kept before each prompt's end | `NEM_SNAPSHOT=1` | off | `--restore-check` with `NEM_CACHE_SWAPS=0`: a difference of 0; then `nem-turns.py`: a new user message reads from the last prompt's end less 4 tokens |
| The swaps' thread | `NEM_HELPER_BG=1` | off | ab.py with the helper, new prompts (48.2 tok/s) and "the helper's swaps" in the token's times (0.4 ms); then `NEM_HELPER_SWAPS=32 NEM_HELPER_EVERY=1`, maya-ling's |
| A prompt's choices on the written tokens' scale | `NEM_CACHE_PROMPT=1`, `2` | off | ab.py's writing after the long prompt (21.7 tok/s against 26.3 on new ones, Q4) |

Not supported yet: pictures and video (the C-RADIO encoder isn't ported), the NextN block's speculative decoding.

The engine's test mode compares against another engine token by token:

```
maya-nem --gguf <...> --ids @prompt.txt --gen 24            greedy, each step's top 5 as id:log-probability
maya-nem --gguf <...> --ids @prompt.txt --follow @their.txt  their tokens fed, each one's log-probability here
```
