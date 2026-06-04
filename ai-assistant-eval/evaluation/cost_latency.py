import os
import time
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

def main():
    # Load .env from the project root
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_path)
    
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(f"HF_TOKEN not found. Looked for .env at {env_path}")
        return
        
    client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=token)
    prompt = "What is the capital of France?"
    
    latencies = []
    print("Running 5 inference calls to Qwen/Qwen2.5-7B-Instruct...")
    for i in range(5):
        start = time.time()
        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
        except Exception as e:
            print(f"Call {i+1} failed: {e}")
            continue
            
        latency = time.time() - start
        latencies.append(latency)
        print(f"Call {i+1} completed in {latency:.2f}s")
        
    if not latencies:
        print("All calls failed.")
        return
        
    avg_latency = sum(latencies) / len(latencies)
    
    # Calculate costs
    input_price = 0.15
    output_price = 0.15
    avg_input_tokens = 50
    avg_output_tokens = 100
    
    # Est. cost per 1K calls
    input_cost_1k = (avg_input_tokens * 1000 / 1_000_000) * input_price
    output_cost_1k = (avg_output_tokens * 1000 / 1_000_000) * output_price
    cost_per_1k = input_cost_1k + output_cost_1k
    
    markdown_table = (
        "| Platform | Model | Avg Latency (s) | Input Price/1M tokens | Output Price/1M tokens | Est. Cost per 1K calls |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        f"| HF Inference API | Qwen/Qwen2.5-7B-Instruct | {avg_latency:.2f} | ${input_price:.2f} | ${output_price:.2f} | ${cost_per_1k:.4f} |\n"
    )
    
    print("\n--- Results ---\n")
    print(markdown_table)
    
    md_path = Path(__file__).resolve().parent / "cost_latency.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_table)
        
    print(f"Saved results to {md_path}")

if __name__ == "__main__":
    main()
