import dimod
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Load data from files
nodes_file_path = r"\task-2-nodes.csv"
adjacency_matrix_file_path = r"\task-2-adjacency_matrix.csv"
nodes_df = pd.read_csv(nodes_file_path)
adjacency_matrix_df = pd.read_csv(adjacency_matrix_file_path)

# Initialize the graph from CSV data
G = nx.Graph()

# Add nodes
for index, row in nodes_df.iterrows():
    node_name = row.iloc[0]  # Extracting the node name
    G.add_node(node_name)

# Add edges with weights from the adjacency matrix
for i, row in adjacency_matrix_df.iterrows():
    node_a = row.iloc[0]
    for j, weight in enumerate(row[1:]):
        node_b = adjacency_matrix_df.columns[j + 1]
        if weight != "-" and not pd.isna(weight):
            weight = float(weight)
            G.add_edge(node_a, node_b, weight=weight)

# Optional visualization
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=500, font_size=8, font_weight="bold")
nx.draw_networkx_edge_labels(
    G, pos, edge_labels={(u, v): f"{d['weight']:.0f}" for u, v, d in G.edges(data=True)}
)
plt.title("Graph Representation of Nodes and Connections")
plt.show()

# Build QUBO
qubo = {}

# Objective function: minimize route cost
for i, j in G.edges:
    qubo[(f"x_{i}_{j}", f"x_{i}_{j}")] = G[i][j]["weight"]

# Constraints to visit each node exactly once
lambda1 = 10  # Penalty for violation
nodes = list(G.nodes)

for node in nodes:
    involved_edges = [
        (f"x_{i}_{node}", f"x_{j}_{node}") for i, j in G.edges if node in (i, j)
    ]
    for u, v in involved_edges:
        qubo[(u, v)] = qubo.get((u, v), 0) - lambda1

# Simulated annealing for QUBO
sampler = dimod.SimulatedAnnealingSampler()
response = sampler.sample_qubo(qubo, num_reads=100)

# Process results
best_solution = response.first.sample
selected_edges = [edge for edge in best_solution if best_solution[edge] == 1]

# Output results
print("Selected routes:")
total_cost = 0

for edge in selected_edges:
    # Extract node numbers from QUBO variable names
    edge_nodes = edge.split("_")[1:]
    u, v = int(edge_nodes[0]), int(edge_nodes[1])

    if G.has_edge(u, v):
        total_cost += G[u][v]["weight"]
        print(f"Route: {u} -> {v}")

print(f"Total cost: {total_cost}")
