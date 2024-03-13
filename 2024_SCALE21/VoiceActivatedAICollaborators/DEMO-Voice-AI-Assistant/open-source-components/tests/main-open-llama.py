import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from datetime import datetime

model_path = "../open_llama_3b"
# model_path = '../open_llama_7b'

before = datetime.now()
print(f"before: {before}")
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="auto",
).to("cpu")

prompt = "Q: What is the largest animal?\nA:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(input_ids=input_ids, max_new_tokens=32)
after = datetime.now()
print(f"after: {after}")

print("\n\n")
difference = after - before
print(f"time: {difference.seconds}")
print("\n\n")
print(tokenizer.decode(generation_output[0]))
