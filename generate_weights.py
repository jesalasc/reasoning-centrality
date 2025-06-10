import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"
path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

inputs = tokenizer("Calculate 13 * 2. Return your answer as an integer.", return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
outputs = model(**inputs, output_attentions=True)
output_attentions = outputs.attentions

def print_weights(attentions, layer=0, batch_i=0, head_i=0):
    print(f"Num layers: {len(attentions)}")
    print(f"Shape vector at layer: {attentions[layer].shape}")
    print(f"Layer {layer}, Batch {batch_i}, Head {head_i}: {attentions[layer][batch_i,head_i]}")

def plot_weights(attentions, layer=0, batch_i=0, head_i=0):
    matrix = attentions[layer][batch_i,head_i]
    flattened_vals = matrix.flatten().detach().cpu().numpy()
    log_transformed = np.log(flattened_vals + 1)
    plt.hist(log_transformed, bins=75)
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

print_weights(output_attentions)
generated = model.generate(**inputs, max_new_tokens=40)
print(f"Output: {tokenizer.decode(generated[0])}")
plot_weights(output_attentions)
