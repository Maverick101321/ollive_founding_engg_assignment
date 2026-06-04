# AI Assistant Evaluation

This project compares an open-source assistant against a frontier assistant, then scores their outputs with an independent LLM judge.

## Setup Instructions

Clone the repository and enter the project directory:

```bash
git clone <repo-url>
cd ai-assistant-eval
```

Create a `.env` file in the parent workspace or project root:

```bash
HF_TOKEN=your_huggingface_token
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the OSS assistant:

```bash
streamlit run oss_assistant/app.py
```

Run the frontier assistant:

```bash
streamlit run frontier_assistant/app.py
```

Run evaluation, scoring, and reporting:

```bash
python evaluation/evaluate.py
python evaluation/score.py
python evaluation/report.py
```

## Architecture Decisions

- **Qwen2.5-7B-Instruct**: used as the open-source model because it is capable enough for general assistant behavior while still being lightweight compared with larger OSS models.
- **Gemini**: used as the frontier model to provide a managed, high-quality baseline with strong instruction following.
- **Llama-3-70B via OpenRouter**: used as an impartial judge so scoring is separate from both evaluated models.
- **Streamlit**: used for the assistant UIs because it is simple, fast to build, and supports chat-style interactions with minimal boilerplate.

## Tradeoffs Made

- Free-tier API limits required pacing and retry logic, especially for Gemini evaluation calls.
- Evaluation is single-turn, which is easier to compare but does not fully test memory or multi-turn behavior.
- The same judge is used across factual, adversarial, and bias categories for consistency, though specialized judges may score each category more precisely.

## What I'd Improve With More Time

- Add multi-turn evaluation scenarios.
- Run automated CI evaluations on prompt or model changes.
- Expand bias prompt diversity across more identities and cultural contexts.
- Deploy the OSS assistant on Hugging Face Spaces for easier review.
