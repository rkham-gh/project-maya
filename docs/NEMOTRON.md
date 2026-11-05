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

## The server

- `serve/frontend.py` knows the template by its `medium_effort`: the server's `low` / `medium` (and `minimal`) turn it
  on, the other levels keep the full default; `"chat_template_kwargs": {"enable_thinking": false}` turns thinking off.
- The server's thinking budget (32,768 tokens unless the run config sets `thinking_budget`) applies: `_sliced` closes
  the reasoning at the first sentence or line end past it. maya-nem ignores the `think_budget` / `think_end` keys of
  `GEN`, the engine's own close for reasoning with no such end.
- A prompt is read on from the engine's state only when it continues the tokens so far (the Mamba-2 state can't be
  rewound). A tool's result does (`test-results\2026-11-04-nemotron\nem-turns.py`: 473 of 511 tokens read on); a new
  user message doesn't, because the template drops the earlier thinking (`truncate_history_thinking`), so the
  conversation is read again from the start. Keeping the state from before each prompt's generation prompt (where the
  next turn's rendering first differs) would save that; not done.
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

Not supported yet: pictures and video (the C-RADIO encoder isn't ported), the NextN block's speculative decoding.

The engine's test mode compares against another engine token by token:

```
maya-nem --gguf <...> --ids @prompt.txt --gen 24            greedy, each step's top 5 as id:log-probability
maya-nem --gguf <...> --ids @prompt.txt --follow @their.txt  their tokens fed, each one's log-probability here
```
