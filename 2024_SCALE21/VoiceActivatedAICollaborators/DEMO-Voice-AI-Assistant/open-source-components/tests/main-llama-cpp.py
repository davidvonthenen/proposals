from llama_cpp import Llama
from datetime import datetime

before = datetime.now()
print(f"before: {before}")
llm = Llama(
    model_path="../models/7B/llama-2-7b-chat.Q4_K_M.gguf",
    # n_gpu_layers=-1, # Uncomment to use GPU acceleration
    # seed=1337, # Uncomment to set a specific seed
    # n_ctx=2048, # Uncomment to increase the context window
)

output = llm(
    "Q: Tell me more about earth. A: ",  # Prompt
    max_tokens=None,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
    stop=[
        "Q:",
        "\n",
    ],  # Stop generating just before the model would generate a new question
    echo=True,  # Echo the prompt back in the output
)  # Generate a completion, can also call create_completion
after = datetime.now()
print(f"after: {after}")

print("\n\n")
difference = after - before
print(f"time: {difference.seconds}")
print("\n\n")

print(output)
