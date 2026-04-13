# Stegosaurus — Demo Day Presentation Outline

**Format**: 15-minute incubator pitch + demo
**Suggested split**: 10 min talk / 3 min live demo / 2 min Q&A buffer

---

## 1. Hook (1 min)

Open with the question: *"What if you could hide a secret message inside a completely normal social media post — one that any AI detector, any moderator, any curious reader would see as innocent text?"*

Show a single slide: an ordinary paragraph of text alongside the decoded message hidden inside it.

---

## 2. The Problem (1.5 min)

- Encrypted messages are obvious — ciphertext looks like noise and signals intent to hide something.
- Steganography (hiding data inside other data) traditionally works on images or audio — not text.
- Text steganography has historically produced unnatural, detectable output.
- There is no accessible, user-friendly tool for hiding messages inside fluent, natural language.

**Who cares?** Journalists, dissidents, privacy advocates, and anyone who wants a practical digital equivalent of invisible ink. Also a genuinely fun toy for social media.

---

## 3. The Solution — What Is Stegosaurus? (2 min)

Stegosaurus is a web app that hides arbitrary secret messages inside AI-generated cover text. The output reads naturally because it *is* natural — every token is chosen by the language model; we just steer *which* valid token gets picked.

Key properties:
- **No shared key required** — encode and decode use the same public model and prompt.
- **Undetectable to casual inspection** — cover text is fluent, on-topic prose.
- **Lossless** — the original message is recovered exactly, bit-for-bit.
- **Open source, self-hostable** — runs on CPU or GPU; ships as a Docker image.

---

## 4. How It Works — The Algorithm (2.5 min)

*Keep this high-level; one diagram is worth more than equations here.*

1. The LLM produces a probability distribution over its vocabulary at each generation step.
2. The top-k tokens are partitioned into equal-probability bins (e.g. 2 bins = 1 bit per token).
3. The next bit(s) of the secret message select which bin to sample from.
4. The highest-probability token in that bin is appended to the cover text.
5. The decoder runs the identical partitioning and reads back which bin each token came from — recovering the bits without any extra metadata.

The approach is a greedy variant of arithmetic coding applied to LLM token distributions (Ziegler et al., 2019 — *Neural Linguistic Steganography*).

**The cover text looks normal because we only choose between tokens the model already considers plausible.** We never force an unlikely word.

---

## 5. Live Demo (3 min)

Walk through the Gradio web UI end-to-end:

1. Enter a prompt and a secret message in the **Encode** tab → show the generated cover text.
2. Copy the cover text to the **Decode** tab (same prompt) → show exact message recovery.
3. Optional: paste the cover text into a search engine or AI detector to show it flags nothing unusual.

*Tip: pre-prepare a prompt and message in case of network issues.*

---

## 6. Architecture & Scalability (2 min)

- **Backend**: Python, PyTorch, Hugging Face Transformers.
- **Frontend**: Gradio, deployable as a single Docker container.
- **Worker pool**: a `WorkerManager` spawns model-serving processes dynamically. On GPU, pool size is capped by VRAM; on CPU, capped by core count with dynamic `torch.set_num_threads()` per worker to prevent thread thrashing.
- **Load tested**: median latency under 10 s at moderate concurrency on CPU; scales horizontally via container replicas.
- **Deployment**: Docker Hub image; documented recipes for GCP Cloud Run, Hugging Face Spaces, and VPS.

---

## 7. Current Status & Roadmap (1.5 min)

**Done:**
- Working encode/decode with Qwen3-0.6B (and configurable for other models).
- Gradio web UI with encode and decode tabs.
- Docker image with documented deployment guides.
- Load testing and CPU/GPU scaling benchmarks.

**Next:**
- Browser extension for one-click encode/decode on social media.
- Support for higher bit-rates (n=4, n=8 bins) to produce shorter cover text.
- Detection-resistance evaluation against statistical steganalysis tools.
- Multi-model support via a model-selector UI.

---

## 8. The Ask (0.5 min)

State clearly what you are looking for from the incubator:

- Mentorship on commercialisation and go-to-market strategy.
- Cloud compute credits for scaling experiments.
- Introductions to privacy/security-focused investors or potential design partners.

---

## Appendix: Slide Suggestions

| Slide | Content |
|---|---|
| 1 | Title + hook — the "before and after" example |
| 2 | The problem — why existing approaches fail |
| 3 | The solution — one-line description + key properties |
| 4 | Algorithm diagram — LLM distribution → bins → token selection |
| 5 | Demo (live or recorded fallback) |
| 6 | Architecture overview — simple block diagram |
| 7 | Traction / benchmarks — latency chart from load tests |
| 8 | Roadmap — 3-column now / next / later |
| 9 | The ask |
