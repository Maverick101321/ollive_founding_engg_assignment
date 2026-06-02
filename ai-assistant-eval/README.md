# AI Assistant Evaluation

This project compares two personal-assistant chatbots:

- `oss_assistant`: Streamlit chatbot using Hugging Face Inference API with `Qwen/Qwen2.5-7B-Instruct`.
- `frontier_assistant`: Streamlit chatbot using Google Gemini API with `gemini-2.5-flash-lite`.

Both assistants use the system prompt:

```text
You are a helpful personal assistant.
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set API keys in your environment:

```bash
export HF_TOKEN="your_huggingface_token"
export GEMINI_API_KEY="your_gemini_api_key"
```

## Run the OSS Assistant

```bash
streamlit run oss_assistant/app.py
```

## Run the Frontier Assistant

```bash
streamlit run frontier_assistant/app.py
```

## Run Evaluation

```bash
python evaluation/evaluate.py
```

The evaluation script writes model responses and latency measurements to `evaluation/results.csv`.
