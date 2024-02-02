from DagmaDCE import utils, nonlinear, nonlinear_dce
import torch, gpytorch
import time
import numpy as np
import matplotlib.pyplot as plt
from CausalDisco.analytics import r2_sortability, var_sortability
from CausalDisco.baselines import r2_sort_regress, var_sort_regress
from cdt.metrics import SID
from scipy.stats import kendalltau
import seaborn as sns
import networkx as nx

sns.set_context("paper")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(device)

torch.set_default_dtype(torch.double)
utils.set_random_seed(0)
torch.manual_seed(0)

reestimate_graph = False
RESULTS_DIR = ''


def create_small_graph(d=5, s0=8, graph_type='ER'):
    """
    Create a small graph for easy visualization.
    Args:
    - d (int): number of nodes in the graph.
    - s0 (int): number of edges to include in the graph.
    - graph_type (str): type of graph, 'ER' for Erdos-Renyi model.
    Returns:
    - B (np.array): adjacency matrix of the generated DAG.
    """
    B = utils.simulate_dag(d, s0, graph_type)
    return B

def visualize_graph(B):
    """
    Visualize the graph given its adjacency matrix.
    Args:
    - B (np.array): adjacency matrix of the DAG.
    """
    G = nx.DiGraph(B)
    layout = nx.spring_layout(G)
    nx.draw(G, layout, with_labels=True, node_color='skyblue', node_size=500, edge_color='black')
    plt.title("Small Test Graph")
    plt.show()

# Create and visualize the small graph
d, s0, graph_type = 5, 8, 'ER'
B_small = create_small_graph(d, s0, graph_type)
visualize_graph(B_small)

# Number of samples
n = 10

# Simulate parameters for the small graph
W_small = utils.simulate_parameter(B_small)

X_small = utils.simulate_linear_sem(W_small, n, sem_type='gauss')

B_true = utils.simulate_dag(d, s0, graph_type)

print(B_true)
print(B_small)