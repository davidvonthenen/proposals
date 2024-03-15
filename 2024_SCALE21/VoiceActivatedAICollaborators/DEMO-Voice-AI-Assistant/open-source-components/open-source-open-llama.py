import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import pyttsx3
from pocketsphinx import LiveSpeech

model_path = "../open_llama_3b"


def main():
    # Speech-to-Text
    print("Listening for speech...")
    for phrase in LiveSpeech():
        if str(phrase) == "exit":
            break
        if str(phrase) == "":
            continue
        if len(str(phrase)) < 5:
            print("(too short)")
            continue

        print(f"Speech-to-Text: {phrase}")

        # LLM
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto",
        ).to("cpu")

        prompt = f"Q: {phrase}\nA:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        generation_output = model.generate(input_ids=input_ids, max_new_tokens=32)

        # Text-to-Speech
        print(f"Text-to-Speech: {tokenizer.decode(generation_output[0])}")

        engine = pyttsx3.init()
        engine.say(tokenizer.decode(generation_output[0]))
        engine.runAndWait()


if __name__ == "__main__":
    main()
