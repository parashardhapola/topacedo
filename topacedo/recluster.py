import networkx as nx
import numpy as np
from tqdm import tqdm


def make_digraph(d: np.ndarray, clust_info=None) -> nx.DiGraph:
    """
    Convert dendrogram into directed graph
    """

    g = nx.DiGraph()
    n = d.shape[0] + 1  # Dendrogram contains one less sample
    if clust_info is not None:
        if len(clust_info) != d.shape[0]+1:
            raise ValueError("ERROR: cluster information doesn't match number of leaves in dendrogram")
    else:
        clust_info = np.ones(d.shape[0]+1)*-1
    for i in tqdm(d, desc='Constructing graph from dendrogram'):
        v = i[2]  # Distance between clusters
        i = i.astype(int)
        g.add_node(n, nleaves=i[3], dist=v)
        if i[0] <= d.shape[0]:
            g.add_node(i[0], nleaves=0, dist=v, cluster=clust_info[i[0]])
        if i[1] <= d.shape[0]:
            g.add_node(i[1], nleaves=0, dist=v, cluster=clust_info[i[1]])
        g.add_edge(n, i[0])
        g.add_edge(n, i[1])
        n += 1
    if g.number_of_edges() != d.shape[0] * 2:
        raise ValueError('ERROR: Number of edges in directed graph not twice the dendrogram shape')
    return g


def find_lca(dag, start_leaf, target_n_leaves):
    tracker = start_leaf
    ancestor = None
    while True:
        p = list(dag.predecessors(tracker))
        if len(p) == 0:
            break
        p = p[0]
        if dag.nodes[p]['nleaves'] == target_n_leaves:
            ancestor = p
            break
        tracker = p
    return ancestor


def get_successors(dag, start):
    q = [start]
    d = []
    while len(q) > 0:
        i = q.pop(0)
        if dag.nodes[i]['nleaves'] == 0:
            d.append(i)
        else:
            q.extend(list(dag.successors(i)))
    return d


def recluster(dag, clust_members, target_n_clusts):
    lca = find_lca(dag, clust_members[0], clust_members.shape[0])
    if lca is None:
        raise ValueError("ERROR: LCA recognition failed")
    q = [lca]
    qvals = [1]
    clust_counts = 1
    clust_nodes = []
    while len(q) > 0:
        pop_index = np.argmax(qvals)
        p, _ = q.pop(pop_index), qvals.pop(pop_index)
        s = list(dag.successors(p))
        for i in s:
            if dag.nodes[i]['nleaves'] > 0:
                qvals.append(dag.nodes[i]['dist'])
                q.append(i)
            else:
                clust_nodes.append([i])
        clust_counts += 1
        if target_n_clusts <= clust_counts:
            break
    if sum([dag.nodes[x]['nleaves'] for x in q])+len(clust_nodes) == len(clust_members) is False:
        raise ValueError("ERROR: Incorrect number of leaves recovered")
    return [get_successors(dag, x) for x in q] + clust_nodes
