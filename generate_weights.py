import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
path = "/Users/canonrobins/Documents/GitHub/DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained(path)

inputs = tokenizer("Given that a equals 2 and b equals 3, Calculate (a * b), give an exact numerical computation", return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
outputs = model(**inputs, output_attentions=True)
output_attentions = outputs.attentions

def print_weights(attentions, layer, batch_i, head_i):
    print(f"Num layers: {len(attentions)}")
    print(f"Shape vector at layer: {attentions[layer].shape}")
    print(f"Layer {layer}, Batch {batch_i}, Head {head_i}: {attentions[layer][batch_i,head_i]}")

print_weights(output_attentions, 0, 0, 0)

generated = model.generate(**inputs, max_new_tokens=400)
print(f"Output: {tokenizer.decode(generated[0])}")
