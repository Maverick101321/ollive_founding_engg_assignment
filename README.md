# Ollive Founding AI/ML Engineer Assignment

This repository contains my submission for the Ollive Founding AI/ML Engineer assignment. The task is to build and evaluate two AI personal assistants:

- an open-source assistant using a Hugging Face model
- a frontier-model assistant using a hosted foundation model API

The assistants must support multi-turn conversation, short-term conversational memory, and basic personal-assistant behavior. The evaluation compares the assistants on factual reliability, jailbreak/content-safety behavior, and bias/fairness handling.

## Approach

The implementation is in [`ai-assistant-eval/`](ai-assistant-eval/).

I built two Streamlit chat applications with the same system prompt:

```text
You are a helpful personal assistant.
```

The OSS assistant uses `Qwen/Qwen2.5-7B-Instruct` through the Hugging Face Inference API. The frontier assistant uses Gemini through the Google Gemini API. Both apps keep chat history in `st.session_state` and render the conversation using Streamlit chat components.

For evaluation, I created a custom prompt set with three categories:

- `factual`: factual questions with some known wrong-answer traps
- `adversarial`: jailbreak and prompt-injection attempts
- `bias`: prompts testing stereotypes, protected attributes, and fairness

The evaluation pipeline runs both models on the same prompts, stores responses in JSON, scores them with a separate LLM judge through OpenRouter, and generates matplotlib charts for comparison.

## Repository Structure

```text
.
├── Founding AI_ML Engineer.pdf
├── README.md
└── ai-assistant-eval/
    ├── oss_assistant/
    │   └── app.py
    ├── frontier_assistant/
    │   └── app.py
    ├── evaluation/
    │   ├── prompts.py
    │   ├── evaluate.py
    │   ├── score.py
    │   ├── report.py
    │   ├── results.json
    │   ├── scores.json
    │   └── charts/
    ├── README.md
    └── requirements.txt
```

## Setup

From the repository root:

```bash
cd ai-assistant-eval
python -m pip install -r requirements.txt
```

Create a `.env` file in the parent folder, next to this README:

```bash
HF_TOKEN=your_huggingface_token
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

The scripts load this `.env` file automatically.

## Run the Assistants

Run the OSS assistant:

```bash
cd ai-assistant-eval
streamlit run oss_assistant/app.py
```

Run the frontier assistant:

```bash
cd ai-assistant-eval
streamlit run frontier_assistant/app.py
```

## Run Evaluation

Generate model responses:

```bash
cd ai-assistant-eval
python evaluation/evaluate.py
```

This writes:

```text
evaluation/results.json
```

Score the responses with the OpenRouter judge:

```bash
python evaluation/score.py
```

This writes:

```text
evaluation/scores.json
```

Generate summary charts:

```bash
python evaluation/report.py
```

This writes charts to:

```text
evaluation/charts/
```

## Evaluation Outputs

The report script generates:

- `bar_chart.png`: grouped category averages for Qwen vs Gemini
- `refusal_rate.png`: adversarial safe-response/refusal rate
- `safety_breakdown.png`: per-prompt adversarial safety scores

It also prints a console summary table of average scores per model per category.

## Notes and Tradeoffs

- The evaluation is single-turn for clean comparison, even though both apps support multi-turn chat.
- API free-tier rate limits required pacing and retry logic, especially for Gemini.
- A single LLM judge is used for consistency across categories.
- The prompt set is custom and compact; with more time, I would expand it with larger public benchmarks and more diverse bias/safety scenarios.

## Further Improvements

- Add multi-turn evaluation cases.
- Add automated CI-based eval runs.
- Deploy the OSS assistant publicly on Hugging Face Spaces.
- Add cost and latency tracking.
- Add observability and guardrail layers.
- Expand memory and tool-use capabilities.
