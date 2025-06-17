import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import scipy
from scipy import stats

device = "mps" if torch.backends.mps.is_available() else "cpu"
path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

inputs = tokenizer("Question: Santiago is the capital city of Chile. What is the capital city of Chile? Answer:", return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
outputs = model(**inputs, output_attentions=True)
output_attentions = outputs.attentions

def print_weights(attentions, layer=0, batch_i=0, head_i=0):
    print(f"Num layers: {len(attentions)}")
    print(f"Shape vector at layer: {attentions[layer].shape}")
    print(f"Layer {layer}, Batch {batch_i}, Head {head_i}: {attentions[layer][batch_i, head_i]}")

def plot_weights(attentions, layer=0, batch_i=0, head_i=0):
    # creating histogram

    layer_matrix = attentions[layer][batch_i]
    flattened_vals = layer_matrix.flatten().detach().cpu().numpy()

    bin_num = int(len(flattened_vals) ** 0.5)
    log_weights = np.log(flattened_vals + 1)
    plt.hist(log_weights, bins=bin_num)
    plt.xlabel("Ln Transformed Weight")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # creating masked relational graph

    total_weights = flattened_vals.shape[0]
    threshold = 1/500 * total_weights

    hist, bin_edges = np.histogram(log_weights, bins=bin_num)
    low_count_bins = np.where(hist < threshold)[0]
    low_ranges = set()
    for i in low_count_bins:
        low_ranges.add((bin_edges[i], bin_edges[i+1]))

    head_matrix = attentions[layer][batch_i, head_i]
    head_matrix_np = head_matrix.detach().cpu().numpy()
    log_matrix = np.log(head_matrix_np + 1)
    mask = np.zeros_like(head_matrix_np, dtype=bool)

    for i, (low, high) in enumerate(low_ranges):
        if high == bin_edges[-1]:
            mask |= (log_matrix >= low) & (log_matrix <= high)
        else:
            mask |= (log_matrix >= low) & (log_matrix < high)

    seq_len = head_matrix.shape[0]
    G = nx.DiGraph()

    for i in range(seq_len):
        G.add_node(i)

    for i in range(seq_len):
        for j in range(seq_len):
            if mask[i, j]:
                weight = head_matrix_np[i, j]
                G.add_edge(i, j, weight=weight)

    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.4e}" for k, v in labels.items()}, font_size=5)
    plt.title(f"Relational Graph at Layer {layer} and Head {head_i}")
    plt.show()

def gof_test(vals):
    # goodness of fit test
    distributions = {"norm": stats.norm, "expon": stats.expon, "uniform": stats.uniform, "lognorm": stats.lognorm,
                     "gamma": stats.gamma, "weibull_min": stats.weibull_min, "pareto": stats.pareto, "genpareto": stats.genpareto,
                     "genexpon": stats.genexpon, "gengamma": stats.gengamma, "t": stats.t}

    adjusted_vals = np.sqrt(vals[(vals > 0.01) & (vals < 0.683)])
    transformed_vals = stats.boxcox(adjusted_vals)[0]
    for dist, fx in distributions.items():
        params = fx.fit(transformed_vals)
        stat, p_value = stats.kstest(transformed_vals, dist, args=params)
        print(f"{dist}: statistic={stat}, p={p_value}\n")

# print_weights(output_attentions)
# generated = model.generate(**inputs, max_new_tokens=150)
# print(f"Output: {tokenizer.decode(generated[0])}")
plot_weights(output_attentions)
