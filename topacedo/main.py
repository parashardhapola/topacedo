import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from scipy.sparse.csgraph import connected_components
import pcst_fast


__all__ = ['pcst', 'calc_neighbourhood_density']



def get_seed_nodes(clusts: pd.Series, frac: float, cff: pd.Series,
                   min_nodes: int, rand_num: int) -> Dict[int, None]:
    seeds = []
    for i in clusts.unique():
        c = clusts[clusts == i]
        if len(c) > min_nodes:
            s = c.sample(frac=min(1, frac+frac*cff[i]), random_state=rand_num).index
            if len(s) < min_nodes:
                s = c.sample(n=min_nodes, random_state=rand_num).index
        else:
            s = c.index
        seeds.extend(s)
    return {x: None for x in seeds}



def pcst(graph, clusters: pd.Series, seed_frac: float, cluster_factor: pd.Series, min_nodes: int,
         rewards: Tuple[float, float], pruning_method: str,
         rand_state: int) -> Tuple[List, List]:

    ss, se = [], []
    _, l = connected_components(graph)
    seeds = get_seed_nodes(clusters, seed_frac, cluster_factor, min_nodes, rand_state)
    print(f"INFO: {len(seeds)} seed cells selected", flush=True)
    for i in set(l):
        idx = np.where(l == i)[0]
        g = graph[idx].T[idx].tocoo()
        c = (1 + g.data.min()) - g.data
        r = [rewards[0] if x in seeds else rewards[1] for x in idx]
        e = np.vstack([g.row, g.col]).T
        x, y = pcst_fast.pcst_fast(e, r, c, -1, 1, pruning_method, 0)
        ss.extend(idx[x])
        se.extend([[idx[x[0]], idx[x[1]]] for x in e[y]])
    cover = set(ss).intersection(list(seeds.keys()))
    if len(cover) != len(seeds):
        print(f"WARNING: Not all seed cells in downsampled data. Try increasing the reward for seeds", flush=True)
    seed_ratio = len(ss) / len(seeds)
    if seed_ratio > 2 and rewards[1] > 0:
        print(f"WARNING: High seed ratio detected. Try decreasing the non-seed reward", flush=True)
    print(f"INFO: {len(ss)} cells sub-sampled. {len(cover)} of which are present in seed list.", flush=True)
    down_ratio = 100 * len(ss)/graph.shape[0]
    down_ratio = "%.2f" % down_ratio
    print(f"INFO: Downsampling percentage: {down_ratio}%", flush=True)
    seed_ratio = "%.3f" % seed_ratio
    print(f"INFO: Seed ratio: {seed_ratio}", flush=True)
    return ss, se



def run_subsampling(self, *, from_assay: str = None, cell_key: str = 'I', feat_key: str = None,
                    cluster_key: str = None, density_key: str = None,
                    min_edge_weight: float = -1, seed_frac: float = 0.05,
                    dynamic_seed_frac: bool = True, dynamic_frac_multiplier: float = 2,
                    min_nodes: int = 3, rewards: tuple = (3, 0.1),
                    rand_state: int = 4466, return_vals: bool = False, label: str = 'sketched'):
    """
    Perform sub-sampling (aka sketching) of cells using the cell-cell neighbourhood graph. Sub-sampling required that
    that cells are partitioned in cluster already. Since, sub-sampling is dependent on cluster information, having,
    large number of homogeneous and even sized cluster improves sub-sampling results.

    Args:
        from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
        cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
        feat_key: Feature key. Should be same as the one that was used in the desired graph. By default the latest
                   used feature for the given assay will be used.
        cluster_key: Name of the column in cell metadata table where cluster information is stored.
        density_key: Name of the column in cell metadata table where neighbourhood density values are stored.
                     Only required if `dynamic_seed_frac` is True.
        min_edge_weight: This parameter is forwarded to `load_graph` and is same as there. (Default value: -1)
        seed_frac: Fraction of cells to be sampled from each cluster. Should be greater than 0 and less than 1.
                   (Default value: 0.05)
        dynamic_seed_frac: if True, then dynamic sampling rate rate will be used. Dynamic sampling takes the mean
                           node density into account while sampling cells from each cluster (default value: True)
        dynamic_frac_multiplier: A scalar value used an multiplier to increase the dynamic sampling rate
        min_nodes: Minimum number of nodes to be sampled from each cluster. (Default value: 3)
        rewards: Reward values for seed and non-seed nodes. A tuple of two values is provided, first for seed nodes
                 and second for non-seed nodes. (Default value: (3, 0.1))
        rand_state: A random values to set seed while sampling cells from a cluster randomly.
        return_vals: If True, then steiner nodes and edges are returned. (Default value: False)
        label: base label for saving values into a cell metadata column (Default value: 'sketched')

    Returns:

    """
    

    if from_assay is None:
        from_assay = self._defaultAssay
    if feat_key is None:
        feat_key = self.get_latest_feat_key(from_assay)
    if cluster_key is None:
        raise ValueError("ERROR: Please provide a value for cluster key")
    clusters = pd.Series(self.cells.fetch(cluster_key, cell_key))
    graph = self.load_graph(from_assay, cell_key, feat_key, 'csr', min_edge_weight, False, False)
    if len(clusters) != graph.shape[0]:
        raise ValueError(f"ERROR: cluster information exists for {len(clusters)} cells while graph has "
                         f"{graph.shape[0]} cells.")
    if dynamic_seed_frac and density_key is None:
        logger.warning("`dynamic_seed_frac` will be ignored because node_density has not been calculated.")
        dynamic_seed_frac = False
    if dynamic_seed_frac:
        if density_key not in self.cells.table:
            raise ValueError(f"ERROR: {density_key} not found in cell metadata table")
        else:
            cff = self.cells.table[self.cells.table.I].groupby(cluster_key)[density_key].median()
            cff = (cff - cff.min()) / (cff.max() - cff.min())
            cff = 1 - cff
            cff = dynamic_frac_multiplier * cff
    else:
        n_clusts = clusters.nunique()
        cff = pd.Series(np.zeros(n_clusts), index=list(range(1, n_clusts + 1)))
    steiner_nodes, steiner_edges = pcst(
        graph=graph, clusters=clusters, seed_frac=seed_frac, cluster_factor=cff, min_nodes=min_nodes,
        rewards=rewards, pruning_method='strong', rand_state=rand_state)
    a = np.zeros(self.cells.table[cell_key].values.sum()).astype(bool)
    a[steiner_nodes] = True

    key = self._col_renamer(from_assay, cell_key, label)
    self.cells.add(key, a, fill_val=False, key=cell_key, overwrite=True)
    logger.info(f"Sketched cells saved with keyname '{key}'")
    if return_vals:
        return steiner_nodes, steiner_edges
