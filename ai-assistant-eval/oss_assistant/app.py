import os

import streamlit as st
from huggingface_hub import InferenceClient


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
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


def visible_messages() -> list[dict[str, str]]:
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

if prompt := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = get_client().chat_completion(
                    messages=st.session_state.messages,
                    max_tokens=512,
                    temperature=0.7,
                )
                assistant_message = response.choices[0].message.content
            except Exception as exc:
                assistant_message = f"Sorry, I ran into an error: {exc}"

            st.markdown(assistant_message)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_message}
    )
