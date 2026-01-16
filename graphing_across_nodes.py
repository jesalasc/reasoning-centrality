import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

np.set_printoptions(formatter={"float": "{:.4f}".format})

device = "mps" if torch.backends.mps.is_available() else "cpu"
path = "google/gemma-2-2b-it"

model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

cent_metrics = {"betweenness": nx.betweenness_centrality,
                "eigenvector": nx.eigenvector_centrality,
                "pagerank": nx.pagerank,
                "degree": nx.degree_centrality,
                "harmonic": nx.harmonic_centrality,
                "katz": nx.katz_centrality}

def average_centrality(inputs, show_graphs=False, show_outputs=False, show_centrality=False, batch_i=0, N=10, cent_metrics_list=None, show_performance=False, return_tensor=False):
    if cent_metrics_list is None:
        cent_metrics_list = ["betweenness", "eigenvector", "pagerank"]

    num_heads = model.config.num_attention_heads
    num_layers = len(model.model.layers)
    centrality_tensor = {metric: np.zeros((num_layers, num_heads, N)) for metric in cent_metrics_list}

    if show_outputs or show_performance:
        output_tracker = []
        none_tracker = 0

    for inp in inputs:
        tokenized_inputs = tokenizer(inp, return_tensors="pt")
        tokenized_inputs = {k: v.to(model.device) for k, v in tokenized_inputs.items()}
        outputs = model(**tokenized_inputs, output_attentions=True)
        output_attentions = outputs.attentions

        for layer in range(num_layers):
            for head_i in range(num_heads):
                head_matrix = output_attentions[layer][batch_i, head_i]
                head_matrix_np = head_matrix.detach().cpu().numpy()

                log_matrix = np.log(head_matrix_np + 1)
                threshold = np.percentile(log_matrix.flatten(), 95)
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

                for metric in cent_metrics_list:
                    G_metric = G.copy()
                    if metric in ["eigenvector", "katz"]:
                        simple_graph = nx.Graph()
                        for u, v, data in G_metric.edges(data=True):
                            weight = data["weight"]
                            if simple_graph.has_edge(u, v):
                                simple_graph[u][v]["weight"] += weight
                            else:
                                simple_graph.add_edge(u, v, weight=weight)
                        G_metric = simple_graph

                    try:
                        node_centralities = cent_metrics[metric](G_metric)
                    except:
                        node_centralities = {i: 0 for i in range(N)}

                    vector = np.array([node_centralities.get(i, 0) for i in range(N)])
                    centrality_tensor[metric][layer, head_i] += vector

        def show_output_func():
            nonlocal none_tracker
            generated = model.generate(**tokenized_inputs, max_new_tokens=50, do_sample=False, temperature=0)
            generated_tokens = generated[0][tokenized_inputs['input_ids'].shape[1]:]
            output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            k_tokens = generated_tokens[:10]
            k_tokens_decoded = tokenizer.decode(k_tokens, skip_special_tokens=True).lower()
            if "yes" in k_tokens_decoded:
                output_tracker.append("yes")
            elif "no" in k_tokens_decoded:
                output_tracker.append("no")
            else:
                output_tracker.append("none")
                none_tracker += 1
            if show_outputs:
                print(f"\n\nInput: {inp}")
                print(f"\n\nOutput: {output}\n\n")

        if show_outputs or show_performance:
            show_output_func()

    for metric in centrality_tensor:
        centrality_tensor[metric] /= len(inputs)

    if return_tensor:
        return centrality_tensor

    return centrality_tensor[cent_metrics_list[0]][0, 0]

def grid_node_head_graphing(centrality_tensor_dict, layer_idx):
    import matplotlib.pyplot as plt

    metrics = list(centrality_tensor_dict.keys())
    _, num_heads, num_nodes = centrality_tensor_dict[metrics[0]].shape

    root = tk.Tk()
    root.title(f"Centrality Graphs for Layer {layer_idx}")

    canvas_frame = tk.Frame(root)
    canvas_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(canvas_frame)
    scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    inner_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner_frame, anchor='nw')

    fig, axes = plt.subplots(num_heads, num_nodes, figsize=(3*num_nodes, 2.5*num_heads), squeeze=False)

    for head in range(num_heads):
        for node in range(num_nodes):
            ax = axes[head][node]
            for metric in metrics:
                y_val = centrality_tensor_dict[metric][layer_idx, head, node]
                ax.plot([layer_idx], [y_val], marker='o', label=metric)
            ax.set_title(f"H{head} N{node}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(fontsize=6)

    plt.tight_layout()
    canvas_fig = FigureCanvasTkAgg(fig, master=inner_frame)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack()

    def resize(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    inner_frame.bind("<Configure>", resize)
    root.mainloop()

inputs = []
words_set = set()
while len(words_set) < 5:
    words = random.choices("abc", weights=[3,1,1], k=5)
    words_set.add(",".join(words))

ans_list = []

for sequence in words_set:
    inputs.append(f"Question: Is 'a' the majority element in the following sequence '{sequence}'. Respond with 'yes' or 'no'.\nAnswer: ")
    if sequence.count("a") >= 3:
        ans_list.append("yes")
    else:
        ans_list.append("no")

centrality_tensor_dict = average_centrality(inputs, N=38, cent_metrics_list=["betweenness", "degree", "katz"], show_performance=False, show_outputs=False, show_graphs=False, return_tensor=True)
grid_node_head_graphing(centrality_tensor_dict, layer_idx=0)
