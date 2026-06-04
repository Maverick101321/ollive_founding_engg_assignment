import json
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
import google.generativeai as genai
from huggingface_hub import InferenceClient

from prompts import EVAL_PROMPTS


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

SYSTEM_PROMPT = "You are a helpful personal assistant."
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
GEMINI_MODEL = "gemini-3.1-flash-lite"
OUTPUT_PATH = Path(__file__).with_name("results.json")


def log(message: str) -> None:
    print(message, flush=True)


def build_qwen_client() -> Optional[InferenceClient]:
    token = os.getenv("HF_TOKEN")
    if not token:
        log("HF_TOKEN is not set. Qwen responses will be marked as skipped.")
        return None

    return InferenceClient(model=QWEN_MODEL, token=token)


def build_gemini_model() -> Optional[genai.GenerativeModel]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log("GEMINI_API_KEY is not set. Gemini responses will be marked as skipped.")
        return None

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )


def run_qwen(client: Optional[InferenceClient], prompt: str) -> str:
    if client is None:
        return "SKIPPED: HF_TOKEN is not set."

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"ERROR: {exc}"


def run_gemini(model: Optional[genai.GenerativeModel], prompt: str) -> str:
    if model is None:
        return "SKIPPED: GEMINI_API_KEY is not set."

    last_error = None

    for attempt in range(3):
        try:
            time.sleep(5)
            response = model.generate_content(prompt)
            return response.text
        except ResourceExhausted as exc:
            last_error = exc
            log("Rate limit hit, waiting 45 seconds...")
            time.sleep(45)
        except Exception as exc:
            return f"ERROR: {exc}"

    return f"ERROR: Gemini rate limit retries exhausted: {last_error}"


def main() -> None:
    qwen_client = build_qwen_client()
    gemini_model = build_gemini_model()
    results = []
    total_prompts = sum(len(prompts) for prompts in EVAL_PROMPTS.values())
    prompt_number = 0

    log(f"Starting evaluation for {total_prompts} prompts.")

    for category, prompts in EVAL_PROMPTS.items():
        log(f"Category: {category} ({len(prompts)} prompts)")

        for prompt in prompts:
            prompt_number += 1
            log(f"[{prompt_number}/{total_prompts}] Running Qwen: {prompt}")
            qwen_response = run_qwen(qwen_client, prompt)

            log(f"[{prompt_number}/{total_prompts}] Running Gemini: {prompt}")
            gemini_response = run_gemini(gemini_model, prompt)

            results.append(
                {
                    "category": category,
                    "prompt": prompt,
                    "qwen_response": qwen_response,
                    "gemini_response": gemini_response,
                }
            )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    log(f"Saved {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
