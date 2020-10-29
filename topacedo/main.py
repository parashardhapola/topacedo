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