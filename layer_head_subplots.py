import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
device = "mps" if torch.backends.mps.is_available() else "cpu"
path = "google/gemma-2-2b-it"
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

# Input text
input_text = "Question: Is 'a' the majority element in the following sequence 'a,a,b,a,c'? Respond with 'yes' or 'no'.\nAnswer: "
tokenized_inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Forward pass
with torch.no_grad():
    outputs = model(**tokenized_inputs, output_attentions=True)
attentions = outputs.attentions

# Parameters
N = 38  # Number of graph nodes
num_layers = len(attentions)
num_heads = attentions[0].shape[1]
seq_len = attentions[0].shape[-1]

# Create figure: smaller per graph, but large scrollable canvas
fig_height = num_heads * 2  # Adjust for compact height
fig_width = num_layers * 2  # Adjust for compact width

fig, axs = plt.subplots(num_heads, num_layers, figsize=(fig_width, fig_height))
plt.subplots_adjust(hspace=0.5, wspace=0.5)

# Plot loop
for layer in range(num_layers):
    for head in range(num_heads):
        attn = attentions[layer][0, head].detach().cpu().numpy()
        log_attn = np.log(attn + 1e-6)
        threshold = np.percentile(log_attn.flatten(), 95)
        mask = log_attn >= threshold

        G = nx.MultiDiGraph()
        token_to_node = {i: i % N for i in range(seq_len)}
        for i in range(N):
            G.add_node(i)

        for i in range(seq_len):
            for j in range(seq_len):
                if mask[i, j]:
                    G.add_edge(token_to_node[i], token_to_node[j], weight=attn[i, j])

        ax = axs[head, layer] if num_heads > 1 else axs[layer]
        ax.set_title(f"L{layer} H{head}", fontsize=6)
        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, node_color="#ADD8E6", edgecolors="black", linewidths=0.4)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, alpha=0.5, width=0.2, arrowsize=3, min_target_margin=1)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=5)
        ax.set_axis_off()

plt.suptitle("Attention Graphs by Layer and Head", fontsize=10)
plt.tight_layout()
plt.savefig("attention_graphs.pdf", bbox_inches="tight")
