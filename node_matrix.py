import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import networkx as nx
import numpy as np
import random
import csv
from collections import defaultdict

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

while len(inputs) < 200:
    letters = random.choices("abc", weights=[3,1,1], k=5)
    sequence = ",".join(letters)
    if sequence in seen:
        continue
    seen.add(sequence)

    inputs.append(f"Question: Is 'a' the majority element in the following sequence '{sequence}'. Respond with 'yes' or 'no'.\nAnswer: ")
    answers.append("yes" if sequence.count("a") >= 3 else "no")

def store_graphs(inputs, answers, layer=0, num_heads=8, N=38):
    stored_graphs = defaultdict(dict)
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

        num_heads = attentions[layer].shape[1]
        for head_i in range(num_heads):
            layer_matrix = attentions[layer][0]
            flattened_vals = layer_matrix.flatten().detach().cpu().numpy()
            log_weights = np.log(flattened_vals + 1)

            # creating masked relational graph based on 95th percentile

            threshold = np.percentile(log_weights, 95)

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

            stored_graphs[input_i][head_i] = G

    return stored_graphs, performance

def compute_centralities(stored_graphs, cent_metrics, num_inputs, num_heads=8, N=38):
    res = {}

    for metric, fx in cent_metrics.items():
        node_matrices = {node: np.zeros((num_heads, num_inputs)) for node in range(N)}

        for input_i in range(num_inputs):
            for head_i in range(len(stored_graphs[input_i])):
                G = stored_graphs[input_i][head_i]
                if metric == "eigenvector" or metric == "katz":
                    simple_graph = nx.Graph()
                    for u, v, data in G.edges(data=True):
                        weight = data["weight"]
                        if simple_graph.has_edge(u, v):
                            simple_graph[u][v]["weight"] += weight
                        else:
                            simple_graph.add_edge(u, v, weight=weight)
                    G = simple_graph

                centrality = fx(G)

                for node in range(N):
                    node_matrices[node][head_i, input_i] = centrality.get(node, 0.1)

        res[metric] = node_matrices

    return res

# def centrality_matrix(inputs, answers, layer=0, num_heads=8, N=38, cent_metric="eigenvector", show_outputs=False):
#     performance = []
#     node_matrices = {node: np.zeros((num_heads, len(inputs))) for node in range(N)}

#     for i, inp in enumerate(inputs):
#         tokenized = tokenizer(inp, return_tensors="pt")
#         tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
#         outputs = model(**tokenized, output_attentions=True)
#         attentions = outputs.attentions

#         generated = model.generate(**tokenized, max_new_tokens=50, do_sample=False)
#         generated_tokens = generated[0][tokenized['input_ids'].shape[1]:]
#         output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

#         k_tokens = generated_tokens[:10]
#         decoded_k = tokenizer.decode(k_tokens, skip_special_tokens=True).lower()

#         if "yes" in decoded_k:
#             prediction = "yes"
#         elif "no" in decoded_k:
#             prediction = "no"
#         else:
#             prediction = "invalid"
#         performance.append(prediction == answers[i])


#         for head_i in range(num_heads):
#             def create_graph():
#                 # exracting layer values

#                 layer_matrix = attentions[layer][0]
#                 flattened_vals = layer_matrix.flatten().detach().cpu().numpy()
#                 log_weights = np.log(flattened_vals + 1)

#                 # creating masked relational graph based on 95th percentile

#                 threshold = np.percentile(log_weights, 95)

#                 head_matrix = attentions[layer][0, head_i]
#                 head_matrix_np = head_matrix.detach().cpu().numpy()
#                 log_matrix = np.log(head_matrix_np + 1)

#                 mask = log_matrix >= threshold

#                 seq_len = head_matrix.shape[0]
#                 G = nx.MultiDiGraph()

#                 token_to_node = {i: i % N for i in range(seq_len)}


#                 for i in range(N):
#                     G.add_node(i)

#                 for i in range(seq_len):
#                     for j in range(seq_len):
#                         if mask[i, j]:
#                             weight = head_matrix_np[i, j]
#                             start_node = token_to_node[i]
#                             end_node = token_to_node[j]
#                             G.add_edge(start_node, end_node, weight=weight)

#                 return G

#             G = create_graph()

#             if cent_metric == "eigenvector" or cent_metric == "katz":
#                 simple_graph = nx.Graph()
#                 for u, v, data in G.edges(data=True):
#                     weight = data["weight"]
#                     if simple_graph.has_edge(u, v):
#                         simple_graph[u][v]["weight"] += weight
#                     else:
#                         simple_graph.add_edge(u, v, weight=weight)
#                 G = simple_graph

#             try:
#                 cent = cent_metrics[cent_metric](G)
#             except:
#                 cent = {n: 0 for n in range(N)}

#             for node in range(N):
#                 node_matrices[node][head_i, i] = cent.get(node, 0.5)

#         if show_outputs:
#             print(f"\nInput: {inp}\nModel Output: {output}\n")

#     return node_matrices, performance

# def display_matrix(matrix, decimals=3, max_samples=None):
#     """Display all matrix values without truncation indicators"""
#     print("\nNode Centrality Matrix (full values):")
#     print("=" * 50)

#     for node, data in matrix.items():
#         print(f"\nNode {node} centrality scores:")
#         for head in range(data.shape[0]):
#             # Print all values for this head
#             values = " ".join([f"{x:.{decimals}f}" for x in data[head]])
#             print(f"  Head {head}: {values}")

# # Usage:
# display_matrix(node_matrix, decimals=3)

# print("\nPerformance Results (True=Correct, False=Incorrect):")
# print("=" * 50)
# for i, correct in enumerate(performance, 1):
#     print(f"Input {i:03d}: {correct}")

def save_node_matrix_to_csv(node_matrices, filename="node_centrality.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "head", "input_index", "centrality"])
        for node, matrix in node_matrices.items():
            num_heads, num_inputs = matrix.shape
            for head in range(num_heads):
                for i in range(num_inputs):
                    writer.writerow([node, head, i, matrix[head, i]])

def save_performance_to_csv(inputs, answers, performance, filename="performance.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input_index", "input_text", "ground_truth", "correct"])
        for i, (text, answer, correct) in enumerate(zip(inputs, answers, performance)):
            writer.writerow([i, text, answer, correct])

# node_matrix, performance = centrality_matrix(inputs, answers)

# save_node_matrix_to_csv(node_matrix, "node_centrality.csv")
# save_performance_to_csv(inputs, answers, performance, "performance.csv")


graphs, performance = store_graphs(inputs, answers)
results = compute_centralities(graphs, cent_metrics, len(inputs))

# Save everything
for metric_name, matrix in results.items():
    save_node_matrix_to_csv(matrix, f"centrality_{metric_name}.csv")

save_performance_to_csv(inputs, answers, performance, "performance.csv")
print(f"\nALL DONE")
