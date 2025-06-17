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

def probability_graph_filter(attentions, layer=0, batch_i=0, head_i=0, N=30):
    # exracting layer values

    layer_matrix = attentions[layer][batch_i]
    flattened_vals = layer_matrix.flatten().detach().cpu().numpy()
    log_weights = np.log(flattened_vals + 1)

    # creating masked relational graph based on 95th percentile

    threshold = np.percentile(log_weights, 95)

    head_matrix = attentions[layer][batch_i, head_i]
    head_matrix_np = head_matrix.detach().cpu().numpy()
    log_matrix = np.log(head_matrix_np + 1)

    mask = log_matrix >= threshold

    seq_len = head_matrix.shape[0]
    G = nx.MultiDiGraph()

    token_to_node = {i: i % N for i in range(seq_len)}


    for i in range(N):
        G.add_node(i)

    for i in range(seq_len):
        for j in range(seq_len):
            if mask[i, j]:
                weight = head_matrix_np[i, j]
                start_node = token_to_node[i]
                end_node = token_to_node[j]
                G.add_edge(start_node, end_node, weight=weight)

    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.4e}" for k, v in labels.items()}, font_size=5)
    plt.title(f"Relational Graph at Layer {layer} and Head {head_i}")
    plt.show()

probability_graph_filter(output_attentions)
