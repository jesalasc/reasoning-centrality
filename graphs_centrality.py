import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random

np.set_printoptions(formatter={"float": "{:.4f}".format})

device = "mps" if torch.backends.mps.is_available() else "cpu"
llama_path = "meta-llama/Llama-3.2-3B-Instruct"
gemma_path = "google/gemma-2-2b-it"
path = gemma_path

model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

cent_metrics = {"betweenness": nx.betweenness_centrality,
                "eigenvector": nx.eigenvector_centrality,
                "pagerank": nx.pagerank,
                "harmonic": nx.harmonic_centrality,
                "katz": nx.katz_centrality}


def average_centrality(inputs, show_graphs=False, show_outputs=False, show_centrality=False, layer=0, batch_i=0, head_i=0, N=10, cent_metric="betweenness", show_performance=False):
    centrality_vector = np.array([0]*N)

    if show_outputs or show_performance:
        output_tracker = []
        none_tracker = 0

    for inp in inputs:
        tokenized_inputs = tokenizer(inp, return_tensors="pt")
        tokenized_inputs = {k: v.to(model.device) for k, v in tokenized_inputs.items()}
        outputs = model(**tokenized_inputs, output_attentions=True)
        output_attentions = outputs.attentions

        # creating graph

        def create_graph(attentions):
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
            nonlocal none_tracker
            generated = model.generate(**tokenized_inputs, max_new_tokens=50, do_sample=False)
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

    def check_accuracy(tracker, answers, none_tracker):
        tracker = np.array(tracker)
        answers = np.array(answers)

        correct = np.sum(answers == tracker)
        return (correct / len(answers), none_tracker / len(answers))

    if show_performance:
        accuracy, unknown = check_accuracy(output_tracker, ans_list, none_tracker)
        print(f"Accuracy is {accuracy}\nRate of unknown performance is {unknown}")

    final_vector = centrality_vector / len(inputs)
    return final_vector

inputs = []
words_set = set()
while len(words_set) < 10:
    words = random.choices("abc", weights=[3,1,1], k=5)
    words_set.add(",".join(words))

ans_list = []

for sequence in words_set:
    inputs.append(f"Question: Is 'a' the majority element in the following sequence '{sequence}'. Respond with 'yes' or 'no'.\nAnswer: ")
    if sequence.count("a") >= 3:
        ans_list.append("yes")
    else:
        ans_list.append("no")


print(f"Average centrality vector: \n{average_centrality(inputs, N=38, cent_metric="katz", show_performance=False, show_outputs=True, show_graphs=False, head_i=5, layer=12)}")
