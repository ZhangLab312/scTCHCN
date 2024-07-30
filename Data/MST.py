import numpy as np
import heapq
import matplotlib.pyplot as plt
import networkx as nx
import scanpy as sc


def maximum_spanning_tree(similarity_matrix):
    num_nodes = similarity_matrix.shape[0]
    max_spanning_tree = []
    max_weight = 0

    # Priority queue to store the edges in the format (weight, node1, node2)
    # We use negative weights because heapq is a min-heap by default
    pq = []
    for j in range(1, num_nodes):
        heapq.heappush(pq, (-similarity_matrix[0][j], 0, j))

    in_tree = [False] * num_nodes
    in_tree[0] = True
    edges_added = 0

    while edges_added < num_nodes - 1:
        # Extract the edge with the maximum weight
        weight, u, v = heapq.heappop(pq)
        if in_tree[v]:
            continue
        in_tree[v] = True
        max_spanning_tree.append((u, v, -weight))
        max_weight += -weight
        edges_added += 1

        # Add new edges to the priority queue
        for w in range(num_nodes):
            if not in_tree[w]:
                heapq.heappush(pq, (-similarity_matrix[v][w], v, w))

    return max_spanning_tree, max_weight


def find_top_related_cells(adata, cell_idx, top_n=15):
    # Extract the expression data for the given cell
    cell_expression = adata.X[cell_idx].toarray().flatten()
    num_cells = adata.X.shape[0]

    # Calculate the Pearson correlation between the given cell and all other cells
    correlations = np.array([
        np.corrcoef(cell_expression, adata.X[i].toarray().flatten())[0, 1]
        for i in range(num_cells)
    ])

    # Find the indices of the top_n most correlated cells
    top_indices = np.argsort(-correlations)[:top_n + 1]

    return top_indices, correlations[top_indices]


def create_correlation_matrix(adata, cell_indices):
    num_cells = len(cell_indices)
    correlation_matrix = np.zeros((num_cells, num_cells))

    for i in range(num_cells):
        for j in range(i, num_cells):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                corr = np.corrcoef(
                    adata.X[cell_indices[i]].toarray().flatten(),
                    adata.X[cell_indices[j]].toarray().flatten()
                )[0, 1]
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr

    return correlation_matrix


# Example usage
if __name__ == "__main__":
    # Load your scRNA-seq data (replace this with your actual data loading code)
    adata = sc.datasets.pbmc3k()  # Example dataset

    # Select a cell index
    cell_idx = 2257  # Replace with your specific cell index

    # Find the top related cells and their correlations
    top_indices, top_correlations = find_top_related_cells(adata, cell_idx)

    # Create the correlation matrix
    correlation_matrix = create_correlation_matrix(adata, top_indices)

    # Compute the maximum spanning tree
    mst, weight = maximum_spanning_tree(correlation_matrix)
    print("Maximum Spanning Tree edges:")
    for u, v, w in mst:
        print(f"{u} -- {v} == {w:.2f}")
    print("Total weight:", weight)

    # Visualization
    G = nx.Graph()
    for u, v, w in mst:
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)  # Seed for reproducibility
    edge_labels = {(u, v): f'{w:.2f}' for u, v, w in mst}

    plt.figure(figsize=(12, 10))

    # Draw the nodes
    node_colors = ["red" if idx == cell_idx else "lightblue" for idx in top_indices]
    labels = {i: top_indices[i] for i in range(len(top_indices))}
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=500, node_color=node_colors, font_size=10,
            font_weight="bold")

    # Draw the edges
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.6)

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Maximum Spanning Tree")
    plt.show()
