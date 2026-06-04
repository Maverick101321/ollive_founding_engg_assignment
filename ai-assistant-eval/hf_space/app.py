import os
import time
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import InferenceClient


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_PROMPT = "You are a helpful personal assistant."


def get_hf_token() -> str | None:
    token = os.getenv("HF_TOKEN")
    if token:
        return token

    try:
        return st.secrets.get("HF_TOKEN")
    except st.errors.StreamlitSecretNotFoundError:
        return None


def get_client() -> InferenceClient:
    return InferenceClient(model=MODEL_ID, token=get_hf_token())


def ensure_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]


def visible_messages() -> list[dict]:
    return [
        message
        for message in st.session_state.messages
        if message["role"] in {"user", "assistant"}
    ]


st.set_page_config(page_title="OSS Assistant")
st.title("OSS Assistant")

ensure_chat_state()

if not get_hf_token():
    st.warning(
        "HF_TOKEN is not set. Add it as an environment variable or in Streamlit secrets."
    )

for message in visible_messages():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "latency" in message:
            st.caption(f"Response time: {message['latency']:.2f}s")

if prompt := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    blocklist = ["bomb", "weapon", "kill", "hack", "exploit"]
    prompt_lower = prompt.lower()
    is_blocked = any(word in prompt_lower for word in blocklist)

    with st.chat_message("assistant"):
        start = time.time()
        
        if is_blocked:
            assistant_message = "I'm not able to help with that."
        else:
            with st.spinner("Thinking..."):
                try:
                    # Filter out 'latency' so it doesn't get sent to the API
                    api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    response = get_client().chat_completion(
                        messages=api_messages,
                        max_tokens=512,
                        temperature=0.7,
                    )
                    assistant_message = response.choices[0].message.content
                except Exception as exc:
                    assistant_message = f"Sorry, I ran into an error: {exc}"
                    
        latency = time.time() - start

        st.markdown(assistant_message)
        st.caption(f"Response time: {latency:.2f}s")

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_message, "latency": latency}
    )
