import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import networkx as nx
import numpy as np
import random
import csv
from collections import defaultdict
import pickle
import os


device = "mps" if torch.backends.mps.is_available() else "cpu"
path = "google/gemma-2-2b-it"

model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="eager", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(path)

cent_metrics = {
    "betweenness": nx.betweenness_centrality,
    "eigenvector": nx.eigenvector_centrality,
    "pagerank": nx.pagerank,
    "harmonic": nx.harmonic_centrality,
    "katz": nx.katz_centrality
}

inputs = []
answers = []
seen = set()


SAVE_PATH = 'abc_inputs.pkl'

def save_data():
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump({
            'inputs': inputs,
            'answers': answers,
        }, f)

def load_data():
    with open(SAVE_PATH, 'rb') as f:
        data = pickle.load(f)
        return data['inputs'], data['answers']

# Uncomment to save
# save_data()

inputs, answers = load_data()


def store_graphs(inputs, answers, num_heads=8, N=38):
    stored_graphs = defaultdict(lambda: defaultdict(dict))
    performance = []

    for input_i, inp in enumerate(inputs):
        # get output

        tokenized = tokenizer(inp, return_tensors="pt")
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
        outputs = model(**tokenized, output_attentions=True)
        attentions = outputs.attentions

        generated = model.generate(**tokenized, max_new_tokens=50, do_sample=False)
        generated_tokens = generated[0][tokenized['input_ids'].shape[1]:]
        output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        k_tokens = generated_tokens[:10]
        decoded_k = tokenizer.decode(k_tokens, skip_special_tokens=True).lower()

        # evaluate prediction

        if "yes" in decoded_k:
            prediction = "yes"
        elif "no" in decoded_k:
            prediction = "no"
        else:
            prediction = "invalid"
        performance.append(prediction == answers[input_i])

        for layer in range(len(attentions)):
            num_heads = attentions[layer].shape[1]
            layer_matrix = attentions[layer][0]
            flattened_vals = layer_matrix.flatten().detach().cpu().numpy()
            log_weights = np.log(flattened_vals + 1)

            # creating masked relational graph based on 95th percentile

            threshold = np.percentile(log_weights, 95)

            for head_i in range(num_heads):
                head_matrix = attentions[layer][0, head_i]
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

                stored_graphs[input_i][layer][head_i] = G

    return stored_graphs, performance

def compute_centralities(stored_graphs, cent_metrics, num_inputs, num_heads=8, N=38, num_layers=26):
    res = {}

    for metric, fx in cent_metrics.items():
        node_matrices = {node: np.zeros((num_layers, num_heads, num_inputs)) for node in range(N)}

        for input_i in range(num_inputs):
            for layer in range(num_layers):
                for head_i in range(len(stored_graphs[input_i][layer])):
                    G = stored_graphs[input_i][layer][head_i]
                    if metric == "eigenvector" or metric == "katz":
                        if metric == "eigenvector":
                            simple_graph = nx.Graph()
                        elif metric == "katz":
                            simple_graph = nx.DiGraph()
                        for u, v, data in G.edges(data=True):
                            weight = data["weight"]
                            if simple_graph.has_edge(u, v):
                                simple_graph[u][v]["weight"] += weight
                            else:
                                simple_graph.add_edge(u, v, weight=weight)
                        G = simple_graph

                        centrality = fx(G, max_iter=10000)
                    else:
                        centrality = fx(G)

                    for node in range(N):
                        node_matrices[node][layer, head_i, input_i] = centrality.get(node, 0.0)

        res[metric] = node_matrices

    return res


def save_node_matrix_to_csv(node_matrices, filename="node_centrality.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "layer", "head", "input_index", "centrality"])
        for node, matrix in node_matrices.items():
            num_layers, num_heads, num_inputs = matrix.shape
            for layer in range(num_layers):
                for head in range(num_heads):
                    for i in range(num_inputs):
                        writer.writerow([node, layer, head, i, matrix[layer, head, i]])

def save_performance_to_csv(inputs, answers, performance, filename="performance.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input_index", "input_text", "ground_truth", "correct"])
        for i, (text, answer, correct) in enumerate(zip(inputs, answers, performance)):
            writer.writerow([i, text, answer, correct])


graphs, performance = store_graphs(inputs, answers)
results = compute_centralities(graphs, cent_metrics, len(inputs))

# Save everything
for metric_name, matrix in results.items():
    save_node_matrix_to_csv(matrix, f"centrality_{metric_name}.csv")

save_performance_to_csv(inputs, answers, performance, "performance.csv")
print(f"\nALL DONE")
