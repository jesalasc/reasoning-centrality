import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

device = "mps" if torch.backends.mps.is_available() else "cpu"
path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

inputs = tokenizer("Question: What is 2 + 2? Answer:", return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
outputs = model(**inputs, output_attentions=True)
output_attentions = outputs.attentions

def print_weights(attentions, layer=0, batch_i=0, head_i=0):
    print(f"Num layers: {len(attentions)}")
    print(f"Shape vector at layer: {attentions[layer].shape}")
    print(f"Layer {layer}, Batch {batch_i}, Head {head_i}: {attentions[layer][batch_i, head_i]}")

def plot_weights(attentions, layer=0, batch_i=0, head_i=0):
    # creating histogram
    matrix = attentions[layer][batch_i,head_i]
    flattened_vals = matrix.flatten().detach().cpu().numpy()
    log_weights = np.log(flattened_vals + 1)
    plt.hist(log_weights, bins=75)
    plt.xlabel("Ln Transformed Weight")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # creating masked relational graph
    total_weights = attentions[layer].shape[2] * attentions[layer].shape[3]
    threshold = 1/2 * total_weights

    hist, bin_edges = np.histogram(log_weights, bins=1000)
    low_count_bins = np.where(hist < threshold)[0]
    bad_ranges = set()
    for i in low_count_bins:
        bad_ranges.add((bin_edges[i], bin_edges[i+1]))

    matrix_np = matrix.detach().cpu().numpy()
    log_matrix = np.log(matrix_np + 1)
    mask = np.zeros_like(matrix_np, dtype=bool)

    for low, high in bad_ranges:
        mask |= (log_matrix >= low) & (log_matrix < high)

    true_mask = ~mask
    print(true_mask)
    seq_len = matrix.shape[0]
    G = nx.DiGraph()

    for i in range(seq_len):
        G.add_node(i)

    for i in range(seq_len):
        for j in range(seq_len):
            if true_mask[i, j]:
                weight = matrix_np[i, j]
                G.add_edge(i, j, weight=weight)

    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: v for k, v in labels.items()})
    plt.title(f"Relational Graph at Layer {layer} and Head {head_i}")
    plt.show()


print_weights(output_attentions)
# generated = model.generate(**inputs, max_new_tokens=150)
# print(f"Output: {tokenizer.decode(generated[0])}")
plot_weights(output_attentions)
