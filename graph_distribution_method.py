import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def average_centrality(inputs, show_graphs=False, show_outputs=False, show_cents=False, layer=0, batch_i=0, head_i=0, N=10):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, attn_implementation="eager").to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    cent_metrics = {"betweenness": nx.betweenness_centrality,
                    "eigenvector": nx.eigenvector_centrality,
                    "pagerank": nx.pagerank,
                    "degree": nx.degree_centrality,
                    "harmonic": nx.harmonic_centrality,
                    "katz": nx.katz_centrality}

    total_centrality = 0

    for inp in inputs:
        tokenized_inputs = tokenizer(inp, return_tensors="pt")
        tokenized_inputs = {k: v.to(model.device) for k, v in tokenized_inputs.items()}
        outputs = model(**tokenized_inputs, output_attentions=True)
        output_attentions = outputs.attentions

        # creating graph

        def create_graph(attentions, layer=layer, batch_i=batch_i, head_i=head_i, N=N, show_graph=show_graphs):
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


            for i in range(min(N, seq_len)):
                G.add_node(i)

            for i in range(seq_len):
                for j in range(seq_len):
                    if mask[i, j]:
                        weight = head_matrix_np[i, j]
                        start_node = token_to_node[i]
                        end_node = token_to_node[j]
                        G.add_edge(start_node, end_node, weight=weight)

            if show_graph:
                pos = nx.circular_layout(G)
                nx.draw(G, pos, with_labels=True)
                labels = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.4e}" for k, v in labels.items()}, font_size=5)
                plt.title(f"Relational Graph at Layer {layer} and Head {head_i}")
                plt.show()

            return G

        graph = create_graph(output_attentions, show_graph=show_graphs)

        # centrality computation

        def compute_centrality(G, metric="betweenness", show_cent=show_cents):
            node_centralities = cent_metrics[metric](G)
            average = sum(node_centralities.values()) / len(node_centralities)
            if show_cent:
                print(f"Average {metric} centrality: {average}")

            return average

        centrality = compute_centrality(graph, metric="pagerank")
        total_centrality += centrality

        # output handling

        def show_output_func():
            generated = model.generate(**inputs, max_new_tokens=150)
            print(f"Output: {tokenizer.decode(generated[0])}")

        if show_outputs:
            show_output_func()

    final_average = total_centrality / len(inputs)
    return final_average
