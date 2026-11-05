# Ling-3.1-flash on Maya (maya-ling)

inclusionAI's Ling-3.1-flash (`Ling-3.1-flash-1019`, MIT) is a 72-layer mixture of experts with linear attention.
`maya-ling` (src/ling) runs it from a GGUF (llama.cpp's `bailingmoe3` architecture; bartowski's
`Ling-3.1-flash-1019-GGUF`) on one NVIDIA GPU and the CPU, behind Maya's own server (serve/server.py), the way maya-ink
runs Inkling-Small (src/ink in the inkling branch, whose structure it follows). llama.cpp's graph
(src/models/bailingmoe3.cpp, delta-net-base.cpp, llama-graph.cpp's build_moe_ffn / build_ffn / build_attn_mha) is the
reference the arithmetic follows; mainline llama.cpp (build 12017) is what the measurements compare against.

## The model

| | |
|---|---|
| Layers | 72 (+ 1 NextN / MTP layer, blk.72, not run). The first 4 have a dense SwiGLU FFN (10,240 wide), the other 68 a MoE |
| Width | 4,096; vocabulary 157,184 (gpt2 BPE, pre-tokenizer `bailingmoe2`, NFC); untied output head |
| Attention, 63 layers | Kimi Delta Attention (KDA): q / k / v projections 4096 -> 64 heads x 128, each a causal depthwise conv of width 4 + SiLU; q and k L2-normed a head; beta = sigmoid(W_beta x) a head; a gate a channel, g = -5 sigmoid(exp(A_log) (W_f x + dt_bias)); a 128 x 128 F32 state a head: S[i][:] *= exp(g_i), d = (v - S^T k) beta, S += k d^T, o = S^T q / sqrt(128); out = rmsnorm(o) w sigmoid(W_g x) a head, then W_o |
| Attention, 9 layers | Gated MLA (layers 7, 15, ..., 71): q = W_qb rmsnorm(W_qa x) (q_lora 1536, 64 heads x (128 nope + 64 rope)); the cache row [rmsnorm(W_kva x)[0:512], rope(...)[512:576]] (MQA, 576 values, F16); scores [k_b q_nope, rope(q_pe)] . row / sqrt(192); a head's 512-value output through v_b (128), times sigmoid(W_gate x) a head; RoPE NORM (consecutive pairs), theta 6e6 |
| MoE | 512 routed experts (1,280 wide) + 1 shared; sigmoid(logits) + exp_probs_b chooses: 8 groups of 64 scored by their top-2 sums, the top 4 groups, the top 8 experts there; weights the chosen sigmoids / their sum x 2.5 |
| SwiGLU clamp | min(silu(g), 10) clamp(u, -10, 10) for the routed experts of layers 65-71 and the shared experts of 64-71 (the GGUF's swiglu_clamp_exp / _shexp): without it layer 64 blows up |
| Context | 262,144 native; the cache is 9 x 576 x 2 bytes = 10.4 KB a token, the KDA state 252 MiB a sequence |

IQ2_XS is 177.4 GB: the routed experts IQ2_XS (up IQ3_XXS / IQ3_S in some layers), 160 GB; attention and the dense
layers Q5_K mostly, 10.2 GB; the shared experts Q8_0, 1.1 GB; output head and embedding Q8_0, 0.7 GB each; the MTP
layer (Q4_0, 4.6 GB) is not loaded.

## The engine

- **The token path** (`Engine::step`): everything but the routed experts on the GPU, int8-activation GEMVs from
  maya-ds4's kernels (src/ds4/ds4_iq.cuh); the router's F32 logits come back to the host, which routes (group-limited).
  The chosen experts in the VRAM cache run on the GPU, the rest on the CPU lane (src/ling/ling_cpu.cpp: maya-ds4's CPU
  rows read the GGUF's mapping in place), at the same time. A MoE layer is one CUDA graph from one host sync to the
  next (the previous layer's end, the attention, the router; `LING_GRAPHS=0`: launches). The KDA recurrence is
  ggml-cuda's layout (a warp 4 value columns of a head, 8 lanes a column, the state in registers); the MLA decode
  attention is flash-decoding over the cache's splits (a block a split and 16 heads), its splits from the position on
  the device, so a graph serves every position.
- **The prompt path** (`Engine::prefill`): chunks of up to 8,192 tokens (`--chunk`) a layer at a time on the GPU; a
  layer's attention in pieces of 2,048 (`--sub`): the dense matrices decoded to F16 and multiplied by cuBLAS with F32
  sums, the KDA recurrence token after token over the piece (the same kernel as the token path), MLA's attention as
  tiles of scores (cuBLAS) and a causal softmax, k_b / v_b as batched GEMMs a head. The router runs in F32. The routed
  experts go in groups of 32 (`--stage`): a group's experts some token chose that are not in the VRAM cache are copied
  into one of two group buffers through the engine's pinned staging, on a copy stream, while the other group is
  multiplied; then each is decoded, multiplied for its tokens and added to theirs times their weights, in id order,
  then the shared expert: the same sums every run. Prompts under 256 new tokens (`LING_MIN_CHUNK`) take the token path:
  a chunk of a few hundred tokens already touches nearly every one of a layer's 512 experts, ~150 GB to copy.
  `LING_PF_PREFETCH=1` (off until measured): a chunk of `LING_PF_PREFETCH_MIN` (4,096) tokens or more uses nearly every
  expert, so a copier thread (its own pinned staging, pool and copy stream) stages each MoE layer's experts that are not
  in the VRAM cache into one of two layer buffers (2 x 3.3 GB in the arena) while the GPU computes the layers before:
  the copies no longer wait for the attention's compute, nor the attention for them. The GEMMs read the same weights in
  the same order (the same bits).
- **The VRAM expert cache**: maya-ink's (slots sized per class of layers' expert bytes, the prompt path's arena lent
  between prompts, the MLA caches' rows past the position lent too; scores decaying 0.99 a token, up to 16 swaps every
  2 tokens at a margin of 1.5, each swap's copy staged in 2 MB pieces while the CPU waits for the GPU; `--profile FILE`
  keeps the scores for the next start). Its settings are maya-ink's measured ones (`LING_CACHE_*`), not yet re-tuned
  for Ling's 512 experts.
- **The CPU lane**: maya-ink's (one spinning pool run a layer: (expert, band of gate / up rows) jobs, then bands of down
  rows summed over the experts), with Ling's clamp form in its epilogue (iq_cpu_gate_up's clamp is DeepSeek's, on g
  before the SiLU: used only where there is no clamp).
- The mapping's pages of the weights copied to VRAM leave the working set after the copy (`LING_TRIM=0`: kept); the
  routed experts' 160 GB are touched once at the start (`LING_PREFAULT=0`: not).
- **A conversation's next turn**: the KDA state cannot go back, so a request is read on from the engine's state only
  when it continues the tokens so far. The state at each prompt's end is also kept aside (`Engine::snapshot`: the KDA
  layers' states and convolutions, 283 MB of VRAM; `LING_SNAPSHOT=0`: not), so a next turn whose last answer the
  template renders differently from the tokens written (its whitespace, its thinking) goes on from the last prompt's
  end - only the answer and the new message are read - instead of the whole conversation from the start.

## Use

```
maya-ling --gguf <...-00001-of-00005.gguf> --serve [--max-context N] [--profile FILE]       Maya's engine protocol
maya-ling --gguf <...> --ids @prompt.txt --gen 24            greedy, each step's top 5 as id:log-probability
maya-ling --gguf <...> --ids @prompt.txt --follow @their.txt  their tokens fed, each one's log-probability here
maya-ling --gguf <...> --runs runs.txt                       several of those with one load
maya-ling --selftest                                         the KDA scan, the MLA decode, the routing vs host references
```

`tools/ling/make_pack.py --gguf <first shard> --out <pack dir>` writes the tokenizer folder the server reads (with the
template the GGUF carries). The server needs no Ling-specific code beyond its turn end: the template is GLM-style
(`<arg_key>` tool calls, `<think>` reasoning, `enable_thinking`), which serve/frontend.py already reads; serve/server.py
stops at `<|role_end|>` (the pack's eos) and `<role>`.

A run config (`--engine maya --config ...`):

```json
{
  "exe": "<fork>\\build-ling\\maya-ling.exe",
  "args": ["--gguf", "<...-00001-of-00005.gguf>", "--max-context", "131072", "--profile", "<pack dir>\\expert-profile.bin"],
  "tokenizer": "<pack dir>\\tokenizer",
  "model_name": "ling-3.1-flash",
  "gpu": [0],
  "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
}
```

## Status

- Written and compiling (CUDA 13.4, sm_120); not yet run on the model. Correctness against llama.cpp and speed: to be
  measured.
