# Inkling 2 Small on Maya (maya-ink, branch inkling2)

Thinking Machines' Inkling 2 Small (weights released 2026-11-12, Apache 2.0) is a 594B mixture of experts with 17B
active per token. This branch turns maya-ink (src/ink, see INKLING.md for the engine itself: the VRAM expert cache, the
helper card, the CPU lane, the prompt path) from Inkling-Small into Inkling 2 Small. It reads a 2-bit GGUF converted from
the released NVFP4 / MXFP8 checkpoint, since nobody had published one; the converter and its checks are in the user's
`C:\llm\test-results\2026-11-12-inkling2` (outside this repository).

## The model

| | |
|---|---|
| Layers | 48, every one a MoE. Layers 5, 11, ..., 47 are global, the other 40 sliding (window 1,024) |
| Width | 4,608. 48 query heads and 8 KV heads of 128 |
| Attention | No RoPE; the relative band (d_rel 16, 1,024 positions) and tau on the global layers as in Inkling-Small. q and k L2-clipped to 22.6 (q times a learned scale), the relative states clipped to 8, v RMS-normed per head, scores / sqrt(128). Gated output (o * sigmoid(gate)) and a sink logit per token and head |
| Block attention residuals | Each layer's input is a softmax mix (a learned query over the RMS-normed states) of the token's finished 6-layer blocks and the running partial; the global layers mix three times, for q, k and v. The output head mixes once more |
| Short convolutions | Static ones as in Inkling-Small, (x + conv) / sqrt(5). The global layers' k and v convolutions are dynamic: static taps plus a projection of the layer's input, / 3 |
| MoE | 384 routed experts (2,304 wide), top 8 of sigmoid(logit) + bias, weights sigmoid / their sum, times 8; no shared experts. SwiGLU softcapped: softcap(g, 4) * sigmoid(g) * softcap(u, 25) |
| Output | Logits softcapped at 30. Vocabulary 201,024, 200,058 real ids |

## The GGUF (arch `inkling2`)

| Part | Type | Size |
|---|---|---|
| Experts' gate and up, layers 0-39 | IQ2_XXS | 84.1 GB |
| Experts' down, layers 0-39 | IQ2_XS | 47.1 GB |
| All experts of layers 40-47 (MXFP8 in the release) | IQ2_S | 31.3 GB |
| Attention, embedding, output head | Q8_0 | 6.9 GB |
| Router, norms, sinks, small tensors | F32 | 0.4 GB |

Quantized with an importance matrix from a reference pass over 303K tokens of the released weights. Teacher-forced on
12,288 held-out positions, this engine picks the released model's top token at 93.9% of them (98.3% where the released
model is more than 50% sure); perplexity 2.449 -> 2.467.

## What changed against Inkling-Small

- ink_model.cu: the arch `inkling2` hyperparameters (Geo: lcap, qkclip, rclip, gcap, ucap, res_block); the layer's new
  tensors (q / k / v input norms, static and dynamic conv weights, residual queries, clips, gate, sinks); the token path
  (tok_attention, tok_router, tok_tail, tok_head) and the prompt path (pf_attention with the sinks in its softmax,
  pf_moe with route2 and the softcapped activation) rewritten for it. Inkling-Small's fused token kernels are off.
- ink_kernels.cu: attn_res / attn_res_tok (the block mixes; the token path's in two wide launches), unorm, head_clip,
  head_unorm, sconv2 / sconv2_chunk, dsconv, attn_decode2 / attn_softmax2 (sinks and gate), route2, act_cap_rows,
  moe_tail2, gate_mul, softcap, add2.
- ink_cpu.cpp: the CPU lane's softcapped SwiGLU.
- tools/strata_tokenizer.py: Inkling's tokenizer.json sets `ignore_merges` (a word whole in the vocabulary is one token);
  it wasn't honored, which split 3 of 159 test texts differently. Inkling-Small's tokenizer has the same flag, so its
  branch has the same bug.
- tools/ink/make_pack.py: the tokenizer folder one level too deep.

## Measured (2026-11-12)

An RTX 5090 at PCIe 5.0 x8 plus an RTX 3090 as the helper card, a Ryzen 9 9950X, 192 GB of DDR5-3600, Windows 11,
CUDA 13.4, 128K context. `ab.py`, each run from the same expert profile:

| | llama.cpp port, experts in RAM | maya-ink, the 5090 alone | maya-ink, both cards |
|---|---|---|---|
| Writing, new prompts | 8.3 tok/s | 19.8 | 23.6 |
| Writing after a 41K-token prompt | 8.5 | 13.4 | 15.0 |
| Reading that prompt | 23 tok/s (59 with 512-token batches) | 601 | 521 |
| A repeated prompt, 3rd time | 8.8 | 22.3 | 28.6 |

About 5,350 of the 18,432 experts fit in the two cards' VRAM (29%), against about 3,130 of 10,240 (31%) for Inkling-Small, and a token takes 8
instead of 6: hence the lower speed. A prompt of a few hundred new tokens still takes ~7 s, because the prompt path
copies nearly every expert the VRAM doesn't hold.

`INK_SNAPSHOT=1` is on in the run config for this model: `--restore-check` (with `INK_CACHE_SWAPS=0`, one card) gives a
difference of 0 over 64 tokens, and agent turns read only their new part (first tokens 7.4-24.6 s against 8.6-38.5).

## Checks

- On 4- and 7-layer cuts of the GGUF, maya-ink and the llama.cpp port both match a PyTorch reference that runs the GGUF
  at 12 of 12 positions; on the full model maya-ink picks llama.cpp's top token at 36 of 37 positions.
- Through the server: the chat checks 6 of 6 (thinking apart, thinking off, a tool call, the answer from its result,
  streaming, JSON) and `derail.py` 0 of 20, with and without the snapshot.
- Thinking off is effort 0 in this template (it has no `enable_thinking`); the server already maps it.
