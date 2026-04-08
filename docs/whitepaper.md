# Stegosaurus: neural linguistic steganography

**A technical overview of the algorithm and its implementation**

---

## 1. Overview

Stegosaurus hides an arbitrary binary message inside naturally-generated text by exploiting the probability distributions produced by a large language model (LLM). At each step of text generation, the model's next-token distribution is partitioned into equal-probability bins. A chunk of the secret message selects which bin to sample from, and the highest-probability token in that bin is appended to the cover text. A recipient with the same model can reverse the process deterministically - no shared key or out-of-band metadata is required.

The approach is a discrete variant of the arithmetic-coding scheme introduced by Ziegler et al. (2019) in *Neural Linguistic Steganography*, simplified to greedy (argmax) sampling within each bin.

---

## 2. Algorithm

### 2.1 Message encoding

A UTF-8 string is serialised to a flat list of bits, MSB-first. An 8-bit end-of-message (EOM) marker - the byte `0xFF` - is appended after the message bits. Because `0xFF` is never a valid UTF-8 byte, it is an unambiguous sentinel regardless of message content. The decoder reads tokens until it sees `0xFF` at a byte boundary; no out-of-band length is required.

```
bits = utf8_bytes_as_bits(message) || 0xFF
```

This costs 8 bits (8 tokens at 1 bit/token), versus 32 bits for a fixed-width length header - a saving of 24 tokens per message.

### 2.2 Top-k partitioning

At each generation step the model produces a probability distribution over its vocabulary. Stegosaurus restricts attention to the top-`k` tokens by probability (default `k = 50`), discarding the long tail.

These `k` candidates are partitioned into `n` bins (default `n = 2`, i.e. one bit per token) using a greedy equal-mass assignment: tokens are considered in descending probability order and placed into whichever bin currently holds the least total probability mass. This balances the bins as well as possible given the discrete token set.

With `n = 2`:

```
partition 0: tokens whose cumulative probability ≈ 0.5
partition 1: tokens whose cumulative probability ≈ 0.5
```

Increasing `n` to 4 or 8 encodes 2 or 3 bits per token, producing shorter cover text at the cost of less freedom in token selection.

### 2.3 Token selection (encode)

The next `log2(n)` bits of the message are read as an integer index `i`. The highest-probability BPE-safe token in partition `i` is appended to the cover text. The extended sequence is fed back as context for the next step.

### 2.4 Bit recovery (decode)

The decoder re-runs the identical partitioning on each cover token in sequence, using the same model and prompt as context. For each cover token it asks: which partition does this token belong to? That partition index re-encodes the original bit chunk. The loop stops when the recovered bit stream ends with `0xFF` at a byte boundary, which is guaranteed to occur exactly once - at the EOM marker.

### 2.5 BPE safety filter

Byte-pair encodings (BPE) can merge adjacent tokens during re-encoding: decoding `[A, B]` to a string and re-encoding it may yield `[C]` rather than `[A, B]`. If this happens, the decoder cannot recover the original token sequence from the cover text string, breaking the round-trip.

Before including a candidate token in any partition, Stegosaurus checks:

```python
decode([prev_id, candidate_id]) → re-encode → == [prev_id, candidate_id]?
```

Tokens that fail this check are silently excluded from all partitions. Because both encoder and decoder run the same filter, partition assignments remain identical.

---

## 3. Implementation

### 3.1 Modules

| File | Role |
|---|---|
| `src/stegosaurus.py` | Core encode/decode logic and CLI |
| `src/model_config.json` | Per-model tokenizer and loading configuration |
| `demo/app.py` | Gradio web interface |

### 3.2 Model configuration

Rather than hardcoding model-specific behaviour, `model_config.json` externalises the parameters that differ across models:

| Field | Purpose |
|---|---|
| `default_prompt` | Seed text; must be identical for encode and decode |
| `dtype` | Tensor dtype (`bfloat16` on GPU, `float32` on CPU) |
| `bpe_check_add_special_tokens` | Whether to prepend BOS during the BPE safety check |
| `cover_add_special_tokens` | Whether to prepend BOS when tokenising the cover text for decode |
| `trust_remote_code` | Passed to `from_pretrained`; `false` for all supported models |

Supported models: `google/gemma-3-1b-pt`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B`, `meta-llama/Llama-3.2-3B`.

### 3.3 Lazy model loading

The model is loaded once per process and cached in module-level variables. Subsequent calls to `encode` or `decode` reuse the same model and tokenizer objects. This is safe for single-threaded use; concurrent calls would require a lock or a worker pool.

### 3.4 Capacity

With default settings (`top_k = 50`, `n_partitions = 2`), each generated token encodes exactly 1 bit. A 40-character ASCII message is 320 bits, plus 8 EOM bits = 328 tokens of cover text. Doubling `n_partitions` to 4 halves the token count; quadrupling to 8 gives 3 bits per token. Capacity scales with `log2(n_partitions)` but naturalness degrades as the model has less freedom to choose high-probability tokens.

---

## 4. Naturalness and detectability

Cover text naturalness depends on how tightly the partition bins are balanced and how large the vocabulary restriction (`top_k`) is. With a well-balanced two-partition split and `top_k = 50`, each selected token is among the model's most probable continuations, so the text reads naturally.

Statistical detectability is not formally analysed here, but the BPE filter and greedy intra-partition selection mean every chosen token is always in the model's top-k, which limits statistical deviation from the model's baseline distribution. An adversary with access to the same model could in principle detect the signal by checking token partition membership, but this requires knowing the model, prompt, and hyperparameters.

---

## 5. Scaling considerations

### 5.1 Throughput bottleneck

Encoding and decoding are inherently sequential: each token requires a forward pass to compute the partition, and the next pass depends on the previous token. There is no mini-batch parallelism across tokens within a single encode/decode job. A 350-token encode on CPU takes ~30 seconds; on a T4 or L4 GPU it takes ~2 seconds.

### 5.2 Request concurrency

Because the model is held in memory and inference is sequential, a single worker process handles one request at a time. For a monolith deployment:

- **CPU**: one core is fully saturated for ~30s per request. Concurrency requires multiple processes, each loading the model independently (high memory cost).
- **GPU**: a single GPU worker handles one request at a time. Additional concurrent requests queue. Memory is occupied by the model (~3 GB for Qwen2.5-1.5B in bfloat16), leaving no room for batch parallelism within a 16 GB GPU anyway for a 1.5B model.

For low traffic, a single worker with a request queue is sufficient. A Gradio demo with `concurrency_count=1` is the simplest correct configuration.

### 5.3 Scaling out

To serve more concurrent users:

| Strategy | How | Tradeoff |
|---|---|---|
| Multiple CPU workers | Run N processes behind a load balancer | N x model RAM (~6 GB float32 each) |
| GPU worker pool | N GPU containers, each with dedicated VRAM | N x GPU cost |
| Async task queue | Celery + Redis; HTTP request returns a job ID, client polls | Extra infrastructure; enables user accounts, history |

The task-queue approach cleanly decouples the web frontend (stateless, cheap) from the GPU workers (stateful, expensive). This is the microservices architecture described in [deployment.md](deployment.md).

### 5.4 Model size vs. latency

Larger models produce more natural-sounding text but are slower and more memory-hungry. The 1.5B Qwen2.5 model is a practical balance for a demo: it fits on a single GPU or in 6 GB of CPU RAM (float32), and token selection quality is visibly better than GPT-2. Moving to a 7B model would require a 24 GB GPU (A10G, L4, or better) and roughly 5x the per-token latency.

### 5.5 Prompt sensitivity

Cover text quality and capacity are both sensitive to the prompt. A short, generic prompt gives the model broad stylistic freedom; a domain-specific prompt can produce more focused output but may reduce the effective vocabulary available for partitioning. The prompt is not part of the cover text and must be shared between encoder and decoder.

---

## 6. References

Ziegler, Z. M., Deng, Y., & Rush, A. M. (2019). *Neural linguistic steganography*. Proceedings of EMNLP-IJCNLP 2019. https://arxiv.org/abs/1909.01501
