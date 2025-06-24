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
N = 38
num_layers = len(attentions)
num_heads = attentions[0].shape[1]
seq_len = attentions[0].shape[-1]
batch_i = 0

fig_height = num_heads * 2  # Adjust for compact height
fig_width = num_layers * 2  # Adjust for compact width

fig, axs = plt.subplots(num_heads, num_layers, figsize=(fig_width, fig_height))
plt.subplots_adjust(hspace=0.5, wspace=0.5)

backbone = True
# Plot loop
for layer in range(num_layers):
    if backbone:
        for head in range(num_heads):
            # retrieving data

            layer_matrix = attentions[layer][batch_i]
            flattened_vals = layer_matrix.flatten().detach().cpu().numpy()

            # creating masked relational graph

            head_matrix = attentions[layer][batch_i, head]
            head_matrix_np = head_matrix.detach().cpu().numpy()

            seq_len = head_matrix.shape[0]
            G = nx.DiGraph()

            token_to_node = {i: i % N for i in range(seq_len)}

            for i in range(min(seq_len, N)):
                G.add_node(i)

            for i in range(seq_len):
                for j in range(seq_len):
                    weight = head_matrix_np[i, j]
                    start_node = token_to_node[i]
                    end_node = token_to_node[j]
                    G.add_edge(start_node, end_node, weight=weight)

            def backbone_filter(G, threshold=0.99):
                backbone = nx.DiGraph()
                for node in G.nodes():
                    connected_nodes = list(G[node])
                    num = len(connected_nodes)
                    if num > 1:
                        s = sum(G[node][neighbor]['weight'] for neighbor in connected_nodes)
                        for neighbor in connected_nodes:
                            p_val = G[node][neighbor]['weight'] / s if s > 0 else 0
                            alpha_val = 1 - (1 - p_val) ** (num - 1) if num > 1 else p_val
                            if alpha_val > threshold:
                                backbone.add_edge(node, neighbor, weight=G[node][neighbor]['weight'])

                return backbone

            G = backbone_filter(G)
            ax = axs[head, layer] if num_heads > 1 else axs[layer]
            ax.set_title(f"L{layer} H{head}", fontsize=6)
            pos = nx.circular_layout(G)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, node_color="#ADD8E6", edgecolors="black", linewidths=0.4)
            nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, alpha=0.5, width=0.2, arrowsize=3, min_target_margin=1)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=5)
            ax.set_axis_off()
    else:
        layer_attn = attentions[layer][0].flatten().detach().cpu().numpy()
        layer_log = np.log(layer_attn + 1)
        threshold = np.percentile(layer_log, 95)
        for head in range(num_heads):
            head_attn = attentions[layer][0, head].detach().cpu().numpy()
            head_log = np.log(head_attn + 1)
            mask = head_log >= threshold

            G = nx.MultiDiGraph()
            token_to_node = {i: i % N for i in range(seq_len)}
            for i in range(N):
                G.add_node(i)

            for i in range(seq_len):
                for j in range(seq_len):
                    if mask[i, j]:
                        G.add_edge(token_to_node[i], token_to_node[j], weight=head_attn[i, j])

            ax = axs[head, layer] if num_heads > 1 else axs[layer]
            ax.set_title(f"L{layer} H{head}", fontsize=6)
            pos = nx.circular_layout(G)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, node_color="#ADD8E6", edgecolors="black", linewidths=0.4)
            nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, alpha=0.5, width=0.2, arrowsize=3, min_target_margin=1)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=5)
            ax.set_axis_off()

plt.suptitle("Attention Graphs by Layer and Head", fontsize=10)
plt.tight_layout()
plt.savefig("attention_graphs_backbone.pdf", bbox_inches="tight")
