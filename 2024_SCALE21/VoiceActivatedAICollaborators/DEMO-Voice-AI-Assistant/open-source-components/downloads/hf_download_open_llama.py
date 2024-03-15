from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
import transformers
import torch

# Enter your local directory you want to store the model in
save_path = "../open_llama_3b"

# Specify the model you want to download from HF
hf_model = "openlm-research/open_llama_3b"

# Instantiate the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    hf_model, return_dict=True, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(hf_model, legacy=False)

# Save the model and the tokenizer in the local directory specified earlier
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
