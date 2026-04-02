
# Stegosaurus: covert social media messaging app

**Dr. George Perdrizet**
*Course capstone project proposal*
*FullStack Academy AI/ML*
*Cohort 2510-FTB-CT-AIM-PT*
**Instructor**: Dr. George Perdrizet
**Teacher assistant**: Andrew Thomas

---

## Section 1: Problem statement

**What problem are you trying to solve or what question are you trying to answer?**

Sending secret messages to friends is fun; kids and adults alike enjoy it. Remember passing secret notes in school? But current methods are obvious: encrypted text looks like gibberish, and secret codes are easy to spot. What if you could hide a message inside a completely normal-looking social media post? This project will create a playful way to exchange hidden messages that anyone scrolling by would never notice.

**Why does this problem matter? Who would benefit from a solution?**

People love sharing inside jokes and secret messages with friends. A tool that lets you post innocent-looking text while secretly communicating with specific friends adds a layer of fun and mystery to social media interactions. It's like a modern digital version of invisible ink or secret decoder rings - a playful way to share private jokes, coordinate surprise parties, or just enjoy the thrill of hiding messages in plain sight.

---

## Section 2: Project objectives

**What is the primary goal of this project?**

Build a web application that lets friends play with hidden messages, encoding secrets within normal-looking social media posts that only their friends can decode.

**List 2-3 specific objectives you will accomplish:**

1. Implement encode/decode functions using arithmetic coding with language model token probabilities
2. Create a simple web interface for encoding secret messages and decoding cover text
3. Demonstrate successful round-trip message encoding and recovery with natural-sounding output

---

## Section 3: Data

**What data will you use?**

No training data needed - the project uses pre-trained language models. A test corpus will be used for evaluating the naturalness of generated text, and potentially a dataset of normal social media posts for detection experiments.

**Data source(s) (provide a link!):**

- Pre-trained GPT-2 model from Hugging Face
  Model: `openai-community/gpt2` (124M parameters) or `openai-community/gpt2-medium` (355M parameters)
  Repository: [https://huggingface.co/openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
  GPT-2 is ideal for this project because it's lightweight enough to run locally, well-documented, and produces coherent text. The smaller variants can run on CPU if needed.
- Social media datasets for naturalness comparison (optional):
    - reddit_tifu: casual Reddit posts, conversational and varied in length - [https://huggingface.co/datasets/reddit_tifu](https://huggingface.co/datasets/reddit_tifu)
    - sentiment140: 1.6M tweets for short-form text comparison - [https://huggingface.co/datasets/stanfordnlp/sentiment140](https://huggingface.co/datasets/stanfordnlp/sentiment140)
    - yelp_review_full: reviews that read like social posts - [https://huggingface.co/datasets/Yelp/yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full)

**Have you verified the data is accessible?**

Yes

**Have you reviewed the data quality and format?**

Yes

**Estimated dataset size:**

Minimal - primarily using pre-trained model weights (~500MB for GPT-2)

**Data limitations or potential issues:**

None significant - the project relies on pre-trained models rather than custom datasets.

**If using external data, list any licensing or usage restrictions:**

GPT-2 is available under MIT license via Hugging Face. Any social media datasets used for evaluation will be properly licensed public datasets.

---

## Section 4: Approach and methods

**Describe your technical approach:**

The core technique is arithmetic coding with language model token probabilities. When a language model predicts the next token, it outputs a probability distribution. We partition this distribution into two groups based on cumulative probability (50% cutoff), then use bits from the secret message to select which partition to sample from. The recipient, using the identical model, can reverse the process by checking which partition each token falls into to recover the hidden bits.

**What tools, libraries, or frameworks will you use?**

- Hugging Face Transformers (GPT-2 or similar)
- PyTorch
- Streamlit or Gradio for web UI
- Reference: Ziegler et al. "Neural Linguistic Steganography" (2019)

**What techniques or algorithms do you plan to apply?**

- Arithmetic coding for message embedding
- Language model probability distributions for token selection
- Binary encoding/decoding of text messages
- Cumulative probability partitioning for bit encoding

---

## Section 5: Expected deliverables

**What will you produce by the end of this project?**

The primary deliverable is a fun, easy-to-use web application built with Streamlit or Gradio that lets users type a secret message and generate a normal-looking post to share with friends, paste a friend's post and reveal the hidden message inside, and see fun stats about how sneaky their message is. This will be accompanied by a Jupyter notebook documenting the analysis and development process, a written report explaining the technical approach and findings, and a presentation demonstrating the project.

*Stretch goals* include letting users pick the vibe of their cover text ("make it sounds like a food review"), support for longer messages, friend groups with shared keys, and fun themes/skins for the interface.

---

## Section 6: Success criteria

**How will you know if your project is successful? What metrics or outcomes will you measure?**

1. Round-trip accuracy: 100% message recovery (secret message → cover text → recovered message)
2. Encoding efficiency: Achieve 2-4 bits per token encoding rate
3. Naturalness: Generated cover text has comparable perplexity to normal model output
4. Usability: Working web interface that demonstrates the complete encode/decode workflow

---

## Section 7: Known risks and challenges

**What obstacles do you anticipate? How will you address them?**

- Encoding efficiency: may need many tokens to hide short messages. Start with proof of concept, optimize later
- Generated text quality: might sound slightly off. Use larger models, add topic conditioning
- Complexity of arithmetic coding implementation: reference implementations exist (Ziegler et al.), can adapt
- Message fragility: modifications break decoding. Document as known limitation, explore error correction as stretch goal

---

## Section 8: Resources needed

**List any resources, access, or support you will need:**

- GPU access for running language models (have two local Pascal cards with 16 GB memory)
- Hugging Face account for model access
- Hugging Face space for live demo with GPU access

---

## Section 9: GitHub repository (recommended)

**Repository URL:** [https://github.com/gperdrizet/stegosarus](https://github.com/gperdrizet/stegosarus)

If not using GitHub, explain your alternative plan for version control and project management:

---

## Advisor feedback

*To be completed by instructor and/or TA*

**Date reviewed:** _3/23/85_____________

**Approval status:**
- [x] Approved
- [ ] Approved with revisions
- [ ] Needs revision

**Comments:**
