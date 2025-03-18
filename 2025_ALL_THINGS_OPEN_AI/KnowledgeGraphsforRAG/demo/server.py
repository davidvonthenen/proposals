import os
from flask import Flask, request, jsonify
from ctransformers import AutoModelForCausalLM

# Adjust this path to point to where your Llama-2-7B-GGUF files are located
MODEL_PATH = "/Users/vonthd/models/neural-chat-7b-v3-3.Q4_K_M.gguf"

# 1) Load LLM on server startup
print("Loading the LLM model... (this might take a while)")
llm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    model_type="llama"  # or "llama2", depending on your ctransformers usage
)
print("LLM model loaded!")

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    """
    Expects JSON like:
    {
      "prompt": "Your combined RAG context + user query...",
      "max_new_tokens": 200
    }
    """
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_new_tokens = data.get("max_new_tokens", 2048)

    # 2) Generate answer
    output = llm_model(prompt, max_new_tokens=max_new_tokens)

    return jsonify({"answer": output})

if __name__ == "__main__":
    # 3) Start the Flask server
    app.run(host="127.0.0.1", port=5000, debug=False)
