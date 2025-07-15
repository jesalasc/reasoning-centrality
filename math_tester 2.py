import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
import gc
import re

np.set_printoptions(formatter={"float": "{:.4f}".format})

def average_centrality(inputs, show_graphs=False, show_outputs=False, show_centrality=False, layer=0, batch_i=0, head_i=0, N=10, cent_metric="betweenness", show_performance=False):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    path = "google/gemma-2-2b-it"

    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, attn_implementation="eager", low_cpu_mem_usage=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    cent_metrics = {"betweenness": nx.betweenness_centrality,
                    "eigenvector": nx.eigenvector_centrality,
                    "pagerank": nx.pagerank,
                    "degree": nx.degree_centrality,
                    "harmonic": nx.harmonic_centrality,
                    "katz": nx.katz_centrality}

    centrality_vector = np.array([0]*N)

    if show_outputs or show_performance:
        output_tracker = []

    for inp in inputs:
        tokenized_inputs = tokenizer(inp, return_tensors="pt")
        tokenized_inputs = {k: v.to(model.device) for k, v in tokenized_inputs.items()}
        outputs = model(**tokenized_inputs, output_attentions=True)
        output_attentions = outputs.attentions

        # creating graph

        def create_graph(attentions):
            # extracting layer values

            layer_matrix = attentions[layer][batch_i]
            flattened_vals = layer_matrix.flatten().detach().cpu().numpy()
            log_weights = np.log(flattened_vals + 1)

            # creating masked relational graph based on 95th percentile

            threshold = np.percentile(log_weights, 95)

            head_matrix = attentions[layer][batch_i, head_i]
            head_matrix_np = head_matrix.detach().cpu().numpy()
            log_matrix = np.log(head_matrix_np + 1)

            mask = log_matrix >= threshold
            attention_layer = model.model.layers[layer].self_attn

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

            if show_graphs:
                pos = nx.circular_layout(G)
                nx.draw(G, pos, with_labels=True)
                labels = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.4e}" for k, v in labels.items()}, font_size=5)
                plt.title(f"Relational Graph at Layer {layer} and Head {head_i}")
                plt.show()

            return G

        graph = create_graph(output_attentions)

        # centrality computation
        if cent_metric == "eigenvector" or cent_metric == "katz":
            simple_graph = nx.Graph()
            for u, v, data in graph.edges(data=True):
                weight = data["weight"]
                if simple_graph.has_edge(u, v):
                    simple_graph[u][v]["weight"] += weight
                else:
                    simple_graph.add_edge(u, v, weight=weight)
            graph = simple_graph

        def compute_centrality(G):
            node_centralities = cent_metrics[cent_metric](G)
            vector = np.array(list(node_centralities.values()))
            vector_len = len(vector)
            centrality_vector_len = len(centrality_vector)
            if vector_len != centrality_vector_len:
                vector = np.append(vector, [0]*(centrality_vector_len - vector_len))
            if show_centrality:
                print(f"Centrality for input {inp}: {vector}")
            return vector

        vector = compute_centrality(graph)
        centrality_vector = centrality_vector + vector

        # output handling

        def show_output_func():
            generated = model.generate(**tokenized_inputs, max_new_tokens=50, do_sample=False, temperature=0)
            generated_tokens = generated[0][tokenized_inputs['input_ids'].shape[1]:]
            output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            k_tokens = generated_tokens[:10]
            k_tokens_decoded = tokenizer.decode(k_tokens, skip_special_tokens=True).lower()
            output_tracker.append(k_tokens_decoded)
            if show_outputs:
                print(f"\n\nInput: {inp}")
                print(f"Output: {output}\n\n")

        if show_outputs or show_performance:
            show_output_func()

    def check_accuracy(tracker, answers):
        tracker = np.array(tracker)
        answers = np.array(answers)

        def is_correct(ans, track):
            pattern = r"(?<!\d)" + re.escape(ans) + r"(?!\d)"
            return bool(re.search(pattern, track))

        correct = np.sum([is_correct(ans, track) for ans, track in zip(answers, tracker)])
        return correct / len(answers)

    if show_performance:
        accuracy = check_accuracy(output_tracker, ans_list)
        print(f"Accuracy is {accuracy}\n")

    final_vector = centrality_vector / len(inputs)
    return final_vector

inputs = []
nums_set = set()
while len(nums_set) < 5:
    nums = random.choices("0123456789", k=3)
    nums_set.add("".join(nums))

ans_list = []

for num in nums_set:
    inputs.append(f"Question: What is the answer to 189 + {num}. Respond with the corresponding three of four digit number.\nAnswer: ")
    ans_list.append(f"{str(189 + int(num))}")


print(f"Average centrality vector: \n{average_centrality(inputs, N=33, cent_metric="katz", show_performance=False, show_outputs=False, show_graphs=True, show_centrality=True)}")
gc.collect()
