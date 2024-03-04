from transformers import AutoTokenizer, GemmaForCausalLM, TextStreamer
import torch


model = GemmaForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

while True:
    print("-----------------------------------------------------")
    input_text = (input("Prompt: "))
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=1000, streamer=TextStreamer(tokenizer))
    print(tokenizer.decode(outputs[0]))
    print("-----------------------------------------------------")