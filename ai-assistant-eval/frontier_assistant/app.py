import os
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL_NAME = "gemini-2.5-flash-lite"
SYSTEM_PROMPT = "You are a helpful personal assistant."


def get_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY")


def ensure_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def to_gemini_history() -> list[dict[str, object]]:
    return [
        {
            "role": message["role"],
            "parts": [message["content"]],
        }
        for message in st.session_state.messages
    ]


def display_role(role: str) -> str:
    return "assistant" if role == "model" else "user"


st.set_page_config(page_title="Frontier Assistant")
st.title("Frontier Assistant")

ensure_chat_state()

api_key = get_api_key()
if not api_key:
    st.warning("GEMINI_API_KEY is not set. Add it to your environment before chatting.")
else:
    genai.configure(api_key=api_key)

for message in st.session_state.messages:
    with st.chat_message(display_role(message["role"])):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if not api_key:
                assistant_message = (
                    "GEMINI_API_KEY is not set. Add it to your environment and try again."
                )
            else:
                try:
                    model = genai.GenerativeModel(
                        MODEL_NAME,
                        system_instruction=SYSTEM_PROMPT,
                    )
                    chat = model.start_chat(history=to_gemini_history()[:-1])
                    response = chat.send_message(prompt)
                    assistant_message = response.text
                except Exception as exc:
                    assistant_message = f"Sorry, I ran into an error: {exc}"

            st.markdown(assistant_message)

    st.session_state.messages.append(
        {"role": "model", "content": assistant_message}
    )
