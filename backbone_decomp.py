import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import networkx as nx

device = "mps" if torch.backends.mps.is_available() else "cpu"
path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

inputs = tokenizer("Question: Santiago is the capital city of Chile. What is the capital city of Chile? Answer:", return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
outputs = model(**inputs, output_attentions=True)
output_attentions = outputs.attentions

def plot_weights(attentions, layer=0, batch_i=0, head_i=0):
    # retrieving data

    layer_matrix = attentions[layer][batch_i]
    flattened_vals = layer_matrix.flatten().detach().cpu().numpy()

    # creating masked relational graph

    total_weights = flattened_vals.shape[0]

    head_matrix = attentions[layer][batch_i, head_i]
    head_matrix_np = head_matrix.detach().cpu().numpy()

    seq_len = head_matrix.shape[0]
    G = nx.DiGraph()

    for i in range(seq_len):
        G.add_node(i)

    for i in range(seq_len):
        for j in range(seq_len):
            weight = head_matrix_np[i, j]
            G.add_edge(i, j, weight=weight)

    def backbone_filter(G, threshold=0.05):
        backbone = nx.Graph()
        for node in G.nodes():
            connected_nodes = list(G[node])
            num = len(connected_nodes)
            if num > 1:
                s = sum(G[node][neighbor]['weight'] for neighbor in connected_nodes)
                for neighbor in connected_nodes:
                    p_val = G[node][neighbor]['weight'] / s
                    alpha_val = 1 - (1 - p_val) ** (num - 1)
                    if alpha_val < threshold:
                        backbone.add_edge(node, neighbor, weight=G[node][neighbor]['weight'])

        return backbone

    backbone_G = backbone_filter(G)

    pos = nx.circular_layout(backbone_G)
    nx.draw(backbone_G, pos, with_labels=True)
    labels = nx.get_edge_attributes(backbone_G, 'weight')
    nx.draw_networkx_edge_labels(backbone_G, pos, edge_labels={k: f"{v:.4e}" for k, v in labels.items()}, font_size=5)
    plt.title(f"Relational Graph at Layer {layer} and Head {head_i}")
    plt.show()
    print(f"Num_edges: {backbone_G.number_of_edges()}")
    print(f"Num_edgesin complete graph: {seq_len * seq_len}")

plot_weights(output_attentions)
