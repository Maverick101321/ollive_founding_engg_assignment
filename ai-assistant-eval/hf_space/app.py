import os
import time
from pathlib import Path

import streamlit as st
from huggingface_hub import InferenceClient

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_PROMPT = "You are a helpful personal assistant."

INPUT_PRICE_PER_M = 0.15
OUTPUT_PRICE_PER_M = 0.15

def get_client() -> InferenceClient:
    # Explicitly pull the HF_TOKEN from the Space's injected environment variables
    return InferenceClient(model=MODEL_ID, token=os.environ.get("HF_TOKEN"))

def ensure_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
    # Initialize observability metrics
    if "total_input_tokens" not in st.session_state:
        st.session_state.total_input_tokens = 0
    if "total_output_tokens" not in st.session_state:
        st.session_state.total_output_tokens = 0
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0

def visible_messages() -> list[dict]:
    return [
        message
        for message in st.session_state.messages
        if message["role"] in {"user", "assistant"}
    ]

st.set_page_config(page_title="OSS Assistant")

# --- Observability Sidebar ---
st.sidebar.title("📊 Observability Metrics")
st.sidebar.metric("Total Cost", f"${st.session_state.get('total_cost', 0):.6f}")
st.sidebar.metric("Input Tokens", int(st.session_state.get('total_input_tokens', 0)))
st.sidebar.metric("Output Tokens", int(st.session_state.get('total_output_tokens', 0)))

st.title("OSS Assistant")

ensure_chat_state()

for message in visible_messages():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "latency" in message:
            st.caption(f"Response time: {message['latency']:.2f}s")

if prompt := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. Safety Guardrail Check
    blocklist = ["bomb", "weapon", "kill", "hack", "exploit"]
    prompt_lower = prompt.lower()
    is_blocked = any(word in prompt_lower for word in blocklist)
    
    # 2. Tool Use Check
    is_weather = "weather" in prompt_lower 

    with st.chat_message("assistant"):
        start = time.time()
        
        if is_blocked:
            assistant_message = "I'm not able to help with that."
        elif is_weather:
            with st.status("Calling Weather API Tool..."):
                time.sleep(1) # Mock API delay
            assistant_message = "The current weather is 72°F and sunny."
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
                    
                    # --- Observability Math ---
                    in_tokens = len(prompt.split()) * 1.33
                    out_tokens = len(assistant_message.split()) * 1.33
                    
                    st.session_state.total_input_tokens += in_tokens
                    st.session_state.total_output_tokens += out_tokens
                    
                    cost = (in_tokens / 1_000_000 * INPUT_PRICE_PER_M) + (out_tokens / 1_000_000 * OUTPUT_PRICE_PER_M)
                    st.session_state.total_cost += cost
                    
                except Exception as exc:
                    assistant_message = f"Sorry, I ran into an error: {exc}"
                    
        latency = time.time() - start

        st.markdown(assistant_message)
        st.caption(f"Response time: {latency:.2f}s")

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_message, "latency": latency}
    )
    
    # Refresh to update the sidebar metrics immediately
    st.rerun()