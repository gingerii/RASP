# main.py

import time

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, coo_matrix, issparse
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import scale
import igraph as ig
from tqdm import tqdm
from multiprocessing import Pool
import glasbey

# NOTE: rpy2 / R ("mclust") are imported lazily inside clustering(method="mclust")
# so the package imports cleanly without R installed. Enable that path with
# `pip install "rasp[mclust]"` (pins rpy2==3.5.16) plus R's `mclust` package.

class RASP:
    @staticmethod
    def reduce(adata, n_pcs=20, n_neighbors=6, beta=2, platform='visium',
               random_state=2024, covariates=None, smooth_covariates=True,
               scale_covariates=False, n_pcs_cov=None,
               key_added='X_pca_smoothed', cov_key='X_pca_cov', copy=False):
        """
        Run Randomized Spatial PCA (RASP) dimensionality reduction.

        This is the core RASP algorithm.

        Stage 1 (always):
          1. Randomized PCA on the (dense) expression matrix ``adata.X``
             (centered, not scaled).
          2. Spatial smoothing of those PC scores with an inverse-distance kNN
             weights matrix ``W`` built from ``adata.obsm['spatial']``.

        Stage 2 (only if ``covariates`` is given): the non-transcriptomic
        covariates are (optionally) spatially smoothed with the same ``W``,
        concatenated to the smoothed stage-1 scores, and a *second* randomized
        PCA is run on the merged matrix to give the final covariate-integrated
        embedding. No further smoothing is applied after the second PCA.

        The input is expected to be already preprocessed/normalized (e.g. the
        DLPFC tutorial uses an SCTransform-corrected matrix).

        Parameters
        ----------
        adata : AnnData
            Expression in ``adata.X`` and spatial coordinates in
            ``adata.obsm['spatial']``.
        n_pcs : int, optional
            Number of principal components for the stage-1 randomized PCA
            (default 20). Clipped to ``min(n_obs, n_vars)``.
        n_neighbors : int, optional
            Number of spatial neighbors used to build the smoothing weights
            (default 6; passed to :meth:`build_weights_matrix`).
        beta : float, optional
            Inverse-distance weighting exponent (default 2; passed to
            :meth:`build_weights_matrix`).
        platform : str, optional
            Spatial platform, controls self-weighting in the weights matrix
            (default 'visium').
        random_state : int, optional
            Seed for the randomized PCA solver(s) (default 2024).
        covariates : array-like, str, or list of str, optional
            Non-transcriptomic covariates to integrate (stage 2). May be an
            ``(n_obs, d)`` array / DataFrame, a single ``adata.obs`` column
            name or ``adata.obsm`` key, or a list of ``adata.obs`` column
            names. If None (default), only stage 1 is run.
        smooth_covariates : bool or sequence of bool, optional
            Whether to spatially smooth the covariates with ``W`` before the
            second PCA (default True). Pass a length-``d`` sequence for
            per-covariate control.
        scale_covariates : bool, optional
            If True, z-score each covariate before concatenation (default
            False; the published method does not require scaling, but it can
            help when covariate magnitudes differ greatly from the PC scores).
        n_pcs_cov : int, optional
            Number of components for the stage-2 PCA (default: same as
            ``n_pcs``). Clipped to the merged matrix dimensions.
        key_added : str, optional
            ``adata.obsm`` key for the smoothed stage-1 scores
            (default 'X_pca_smoothed'). This is the final embedding when no
            covariates are supplied.
        cov_key : str, optional
            ``adata.obsm`` key for the stage-2 covariate-integrated embedding
            (default 'X_pca_cov'); only written when ``covariates`` is given.
        copy : bool, optional
            If True, operate on and return a copy; otherwise modify ``adata``
            in place (default False).

        Returns
        -------
        adata : AnnData
            Updated with:
              - ``adata.obsm[key_added]`` : smoothed stage-1 scores (n_obs x n_pcs)
              - ``adata.obsm['X_pca']``   : unsmoothed stage-1 scores
              - ``adata.varm['RASP_PCs']``: stage-1 loadings (n_vars x n_pcs)
              - ``adata.obsm[cov_key]``   : stage-2 embedding (only with covariates)
              - ``adata.uns['RASP']``     : run parameters, explained variance,
                and ``embedding_key`` naming the final embedding to use downstream.

        Notes
        -----
        Downstream steps (neighbors/clustering) should use the embedding named
        by ``adata.uns['RASP']['embedding_key']`` — ``key_added`` for stage-1
        only, or ``cov_key`` when covariates were integrated.

        Gene reconstruction operates on the stage-1 result::

            RASP.reduce(adata, n_pcs=20)
            RASP.reconstruct_gene(adata, adata.obsm['X_pca_smoothed'],
                                  adata.varm['RASP_PCs'].T, gene_name='TMSB10')
        """
        adata = adata.copy() if copy else adata

        if 'spatial' not in adata.obsm:
            raise KeyError(
                "adata.obsm['spatial'] not found; RASP requires 2D spatial "
                "coordinates in adata.obsm['spatial']."
            )

        # --- Stage 1: randomized PCA on the (dense) expression matrix. -------
        data = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
        n_pcs = int(min(n_pcs, min(data.shape)))
        pca = PCA(n_components=n_pcs, svd_solver='randomized',
                  random_state=random_state)
        pca_scores = pca.fit_transform(data)

        # Inverse-distance spatial smoothing of the stage-1 PC scores.
        weights = RASP.build_weights_matrix(
            adata, n_neighbors=n_neighbors, beta=beta, platform=platform)
        smoothed = weights @ csr_matrix(pca_scores)
        smoothed = smoothed.toarray() if issparse(smoothed) else np.asarray(smoothed)

        adata.obsm[key_added] = smoothed
        adata.obsm['X_pca'] = pca_scores
        adata.varm['RASP_PCs'] = pca.components_.T

        uns = {
            'params': {
                'n_pcs': n_pcs,
                'n_neighbors': n_neighbors,
                'beta': beta,
                'platform': platform,
                'random_state': random_state,
                'key_added': key_added,
            },
            'variance_ratio': pca.explained_variance_ratio_,
            'n_stages': 1,
            'embedding_key': key_added,
        }

        # --- Stage 2: covariate integration via a second randomized PCA. -----
        if covariates is not None:
            cov, cov_names = RASP._resolve_covariates(adata, covariates)

            if scale_covariates:
                std = cov.std(axis=0)
                std[std == 0] = 1.0
                cov = (cov - cov.mean(axis=0)) / std

            # Optional per-covariate spatial smoothing with the same W.
            smooth_mask = smooth_covariates
            if np.isscalar(smooth_mask) or isinstance(smooth_mask, bool):
                smooth_mask = np.full(cov.shape[1], bool(smooth_mask))
            else:
                smooth_mask = np.asarray(smooth_mask, dtype=bool)
                if smooth_mask.shape[0] != cov.shape[1]:
                    raise ValueError(
                        "smooth_covariates sequence length "
                        f"({smooth_mask.shape[0]}) must match the number of "
                        f"covariates ({cov.shape[1]})."
                    )
            cov_s = cov.copy()
            if smooth_mask.any():
                cov_s[:, smooth_mask] = np.asarray(
                    weights @ cov[:, smooth_mask])

            # Concatenate smoothed stage-1 scores with (smoothed) covariates
            # and run the stage-2 randomized PCA.
            merged = np.hstack([smoothed, cov_s])
            p2 = int(min(n_pcs_cov or n_pcs, min(merged.shape)))
            pca2 = PCA(n_components=p2, svd_solver='randomized',
                       random_state=random_state)
            cov_embedding = pca2.fit_transform(merged)

            adata.obsm[cov_key] = cov_embedding
            uns['n_stages'] = 2
            uns['embedding_key'] = cov_key
            uns['params'].update({
                'covariate_names': cov_names,
                'smooth_covariates': smooth_mask.tolist(),
                'scale_covariates': bool(scale_covariates),
                'n_pcs_cov': p2,
                'cov_key': cov_key,
            })
            uns['stage2_variance_ratio'] = pca2.explained_variance_ratio_

        adata.uns['RASP'] = uns
        return adata

    @staticmethod
    def _resolve_covariates(adata, covariates):
        """
        Coerce the ``covariates`` argument to a float ``(n_obs, d)`` array.

        Accepts an array / DataFrame, a single ``adata.obs`` column name or
        ``adata.obsm`` key, or a list of ``adata.obs`` column names. Returns
        ``(array, names)``.
        """
        # String -> obs column or obsm key.
        if isinstance(covariates, str):
            if covariates in adata.obs.columns:
                arr = adata.obs[[covariates]].to_numpy()
                names = [covariates]
            elif covariates in adata.obsm:
                arr = np.asarray(adata.obsm[covariates])
                if arr.ndim == 1:
                    arr = arr[:, None]
                names = [f"{covariates}_{i}" for i in range(arr.shape[1])]
            else:
                raise KeyError(
                    f"covariate '{covariates}' not found in adata.obs columns "
                    "or adata.obsm keys."
                )
        # List/tuple of obs column names.
        elif isinstance(covariates, (list, tuple)) and covariates and \
                all(isinstance(c, str) for c in covariates):
            missing = [c for c in covariates if c not in adata.obs.columns]
            if missing:
                raise KeyError(f"covariate columns not in adata.obs: {missing}")
            arr = adata.obs[list(covariates)].to_numpy()
            names = list(covariates)
        # pandas DataFrame.
        elif isinstance(covariates, pd.DataFrame):
            arr = covariates.to_numpy()
            names = list(covariates.columns.astype(str))
        # Plain array-like.
        else:
            arr = np.asarray(covariates)
            if arr.ndim == 1:
                arr = arr[:, None]
            names = [f"covariate_{i}" for i in range(arr.shape[1])]

        arr = np.asarray(arr, dtype=float)
        if arr.shape[0] != adata.n_obs:
            raise ValueError(
                f"covariates have {arr.shape[0]} rows but adata has "
                f"{adata.n_obs} observations."
            )
        if not np.isfinite(arr).all():
            raise ValueError("covariates contain non-finite values (NaN/inf).")
        return arr, names

    @staticmethod
    def build_weights_matrix(adata, n_neighbors=6, beta=2, platform='visium'):
        """
        Build a sparse distance matrix including only the K nearest neighbors, and compute inverse weighting.

        Parameters:
        - adata: Annotated data object.
        - n_neighbors: int - number of nearest neighbors to include.
        - beta: weight exponent parameter.
        - platform: string - type of platform.

        Returns:
        - sparse_distance_matrix: csr_matrix - sparse distance matrix of shape (n_samples, n_samples).
        """
        coords = adata.obsm['spatial']
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        # Build the sparse matrix
        data = distances.flatten()
        row_indices = np.repeat(np.arange(coords.shape[0]), n_neighbors)
        col_indices = indices.flatten()
        sparse_distance_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(coords.shape[0], coords.shape[0])).tocsr()

        # Remove outliers
        temp_matrix = sparse_distance_matrix.tocoo()
        percentile_99 = np.percentile(temp_matrix.data, 99)
        temp_matrix.data[temp_matrix.data > percentile_99] = 0
        sparse_distance_matrix = temp_matrix.tocsr()

        # Invert and exponentiate non-zero values
        non_zero_values = sparse_distance_matrix.data[sparse_distance_matrix.data > 0]
        min_non_zero_value = np.min(non_zero_values) if non_zero_values.size > 0 else 1

        if platform == 'visium':
            sparse_distance_matrix.setdiag(min_non_zero_value / 2)
        else:
            sparse_distance_matrix.setdiag(min_non_zero_value)

        inverse_sq_data = np.zeros_like(sparse_distance_matrix.data)
        inverse_sq_data[sparse_distance_matrix.data > 0] = 1 / (sparse_distance_matrix.data[sparse_distance_matrix.data > 0] ** beta)

        inverse_sq_matrix = csr_matrix((inverse_sq_data, sparse_distance_matrix.indices, sparse_distance_matrix.indptr),
                                        shape=sparse_distance_matrix.shape)

        row_sums = np.asarray(inverse_sq_matrix.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        weights = inverse_sq_matrix.multiply(1 / row_sums[:, np.newaxis])

        return weights

    @staticmethod
    def clustering(adata, n_clusters=7, n_neighbors=10, key='X_pca_smoothed', method='mclust'):
        """
        Spatial clustering.

        Parameters:
        - adata: AnnData object of scanpy package.
        - n_clusters: int, optional - The number of clusters. Default is 7.
        - n_neighbors: int, optional - The number of neighbors considered during refinement. Default is 15.
        - key: string, optional - The key of the learned representation in adata.obsm. Default is 'X_pca_smoothed'.
        - method: string, optional - The tool for clustering. Supported tools: 'mclust', 'leiden', 'louvain'.

        Returns:
        - adata: Updated AnnData object with clustering results.
        """

        if method == 'mclust':
            np.random.seed(2020)
            try:
                import rpy2.robjects as robjects
                import rpy2.robjects.numpy2ri
            except ImportError as e:
                raise ImportError(
                    "method='mclust' requires rpy2 and an R installation with the "
                    "'mclust' package. Install with `pip install \"rasp[mclust]\"` "
                    "(pins rpy2==3.5.16) and, in R, run install.packages('mclust'). "
                    "Otherwise choose method='louvain', 'leiden', 'KMeans', or "
                    "'walktrap', which have no R dependency."
                ) from e
            robjects.r.library("mclust")
            rpy2.robjects.numpy2ri.activate()
            r_random_seed = robjects.r['set.seed']
            r_random_seed(2020)
            rmclust = robjects.r['Mclust']
            res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[key]), n_clusters, 'EEE')
            mclust_res = np.array(res[-2])
            adata.obs[f'RASP_{method}_clusters'] = mclust_res
            adata.obs[f'RASP_{method}_clusters'] = adata.obs[f'RASP_{method}_clusters'].astype('int')
            adata.obs[f'RASP_{method}_clusters'] = adata.obs[f'RASP_{method}_clusters'].astype('category')

        elif method == 'louvain':
            adata = RASP.louvain(adata, n_clusters, n_neighbors=n_neighbors, key_added='RASP_louvain_clusters')

        elif method == 'leiden':
            adata = RASP.leiden(adata, n_clusters, n_neighbors=n_neighbors, key_added='RASP_leiden_clusters')

        elif method =="walktrap":
                neighbors_graph = adata.obsp['connectivities']
                sources, targets = neighbors_graph.nonzero()
                weights = np.asarray(neighbors_graph[sources, targets]).ravel()
                g = ig.Graph(directed=False)
                g.add_vertices(adata.n_obs)
                g.add_edges(zip(sources, targets))
                g.es['weight'] = weights
            
                # Perform Walktrap community detection
                start_time = time.time()
                walktrap = g.community_walktrap(weights='weight')
                clusters = walktrap.as_clustering(n=n_clusters)
                end_time = time.time()
                cluster_time = end_time - start_time
                adata.obs[f'RASP_{method}_clusters'] = pd.Categorical(clusters.membership)

        elif method == "KMeans":
            kmeans = KMeans(n_clusters = n_clusters,random_state = 10)
            adata.obs[f'RASP_{method}_clusters'] = pd.Categorical(kmeans.fit_predict(adata.obsm['X_pca_smoothed']))

        num_clusters = len(set(adata.obs[f'RASP_{method}_clusters']))
        palette = glasbey.create_palette(palette_size=num_clusters)
        adata.uns[f'RASP_{method}_clusters_colors'] = palette

        return adata

    @staticmethod
    def louvain(adata,n_clusters,n_neighbors = 10,use_rep = 'X_pca_smoothed',
                key_added = 'RASP_louvain_clusters',random_seed = 2023):
        """
        Perform Louvain clustering on the AnnData object.

        Parameters:
        - adata: AnnData object of the scanpy package.
        - n_clusters: int - The desired number of clusters.
        - n_neighbors: int, optional - The number of neighbors to consider (default is 10).
        - use_rep: str, optional - The representation to use for clustering (default is 'X_pca_smoothed').
        - key_added: str, optional - Key for storing the clustering results in adata.obs (default is 'RASP_louvain_clusters').
        - random_seed: int, optional - Random seed for reproducibility (default is 2023).

        Returns:
        - adata: Updated AnnData object with Louvain clustering results.
        """
        res = RASP.res_search_fixed_clus_louvain(
            adata, 
            n_clusters, 
            increment=0.1, 
            start = 0.001,
            random_seed=random_seed)
        
        print(f'resolution is: {res}')
        sc.tl.louvain(adata, random_state=random_seed, resolution=res)
       
        adata.obs[key_added] = adata.obs['louvain']
        adata.obs[key_added] = adata.obs[key_added].astype('int')
        adata.obs[key_added] = adata.obs[key_added].astype('category')

        return adata

    @staticmethod
    def leiden(adata,n_clusters,n_neighbors = 10,use_rep = 'X_pca_smoothed',
               key_added = 'RASP_leiden_clusters',random_seed = 2023):
        """
        Perform Leiden clustering on the AnnData object.

        Parameters:
        - adata: AnnData object of the scanpy package.
        - n_clusters: int - The desired number of clusters.
        - n_neighbors: int, optional - The number of neighbors to consider (default is 10).
        - use_rep: str, optional - The representation to use for clustering (default is 'X_pca_smoothed').
        - key_added: str, optional - Key for storing the clustering results in adata.obs (default is 'RASP_leiden_clusters').
        - random_seed: int, optional - Random seed for reproducibility (default is 2023).

        Returns:
        - adata: Updated AnnData object with Leiden clustering results.
        """
        
        res = RASP.res_search_fixed_clus_leiden(
            adata, 
            n_clusters, 
            increment=0.1, 
            start = 0.001,
            random_seed=random_seed)
        
        print(f'resolution is: {res}')
        sc.tl.leiden(adata, random_state=random_seed, resolution=res)
       
        adata.obs[key_added] = adata.obs['leiden']
        adata.obs[key_added] = adata.obs[key_added].astype('int')
        adata.obs[key_added] = adata.obs[key_added].astype('category')

        return adata

    @staticmethod
    def res_search_fixed_clus_louvain(adata, n_clusters, increment=0.1, start=0.001, random_seed=2023):
        """
        Search for the correct resolution for the Louvain clustering algorithm.

        Parameters:
        - adata: AnnData object containing the data.
        - n_clusters: int - The target number of clusters.
        - increment: float, optional - The step size for resolution search (default is 0.1).
        - start: float, optional - The starting resolution for the search (default is 0.001).
        - random_seed: int, optional - Random seed for reproducibility (default is 2023).

        Returns:
        - float: The largest correct resolution found for the specified number of clusters.
        """
        if increment < 0.0001:
            print("Increment too small, returning starting value.")
            return start  # Return the initial start value
        #keep track of the currect resolution and the largest resolution that is not to large. 
        largest_correct_res = None
        current_res = start
        for res in np.arange(start,2,increment):
            sc.tl.louvain(adata,random_state = random_seed,resolution = res)
            
            #increase res tracker to current res
            current_res = res

            
            num_clusters = len(adata.obs['louvain'].unique())
            print(f'Resolution: {res} gives cluster number: {num_clusters}')

            if num_clusters == n_clusters:
                largest_correct_res = res  # Update the largest correct resolution found
            
            #Check to see if the res resulted in too many clusters! 
            #break out of loop if we exceed this point. 
            if num_clusters > n_clusters:
                break

        
        #return correct res if you have one! 
        if largest_correct_res is not None:
            return largest_correct_res

        #perform tail end recursion until correct res is found! 
        else:
            return RASP.res_search_fixed_clus_louvain(
                adata,
                n_clusters,
                increment = increment/10,
                start = current_res - increment,
                random_seed = random_seed)


    @staticmethod
    def res_search_fixed_clus_leiden(adata, n_clusters, increment=0.1, start=0.001, random_seed=2023):
        """
        Search for the correct resolution for the Leiden clustering algorithm.

        Parameters:
        - adata: AnnData object containing the data.
        - n_clusters: int - The target number of clusters.
        - increment: float, optional - The step size for resolution search (default is 0.1).
        - start: float, optional - The starting resolution for the search (default is 0.001).
        - random_seed: int, optional - Random seed for reproducibility (default is 2023).

        Returns:
        - float: The largest correct resolution found for the specified number of clusters.
        """
        if increment < 0.0001:
            print("Increment too small, returning starting value.")
            return start  # Return the initial start value
        #keep track of the currect resolution and the largest resolution that is not to large. 
        largest_correct_res = None
        current_res = start
        for res in np.arange(start,2,increment):
            sc.tl.leiden(adata,random_state = random_seed,resolution = res)
            
            #increase res tracker to current res
            current_res = res

            
            num_clusters = len(adata.obs['leiden'].unique())
            print(f'Resolution: {res} gives cluster number: {num_clusters}')

            if num_clusters == n_clusters:
                largest_correct_res = res  # Update the largest correct resolution found
            
            #now check to see if the res resulted in too many clusters! 
            #break out of loop if we exceed this point. 
            if num_clusters > n_clusters:
                break

        
        #return correct res if you have one! 
        if largest_correct_res is not None:
            return largest_correct_res

        #perform tail end recursion until correct res is found! 
        else:
            return RASP.res_search_fixed_clus_leiden(
                adata,
                n_clusters,
                increment = increment/10,
                start = current_res - increment,
                random_seed = random_seed)

    @staticmethod
    def fx_1NN(index, location_in):
        """
        Python equivalent of the fx_1NN function that is called in the loop.
        Computes the distance from the point at 'index' to its nearest neighbor.
        """
        distances = cdist([location_in[index]], location_in, 'euclidean')
        nearest_neighbor = np.partition(distances, 1)[0, 1]  # 1st closest distance
        return nearest_neighbor
    @staticmethod

    
    def CHAOS(clusterlabel, location):
        """
        Calculate the CHAOS score, quantifying the spatial continuity and compactness of clusters.

        Parameters:
        - clusterlabel: array-like - An array representing the cluster labels for each data point.
        - location: array-like - An array of spatial coordinates corresponding to each data point.

        Returns:
        - float: The calculated CHAOS score representing spatial coherence of the clusters.
        """
        
        matched_location = np.array(location)
        clusterlabel = np.array(clusterlabel)
        
        # Remove NA (None) values
        NAs = np.where(pd.isna(clusterlabel))[0]
        if len(NAs) > 0:
            clusterlabel = np.delete(clusterlabel, NAs)
            matched_location = np.delete(matched_location, NAs, axis=0)
    
        # Standardize the location data
        matched_location = scale(matched_location)
    
        unique_labels = np.unique(clusterlabel)
        dist_val = np.zeros(len(unique_labels))
        
        for count, k in enumerate(unique_labels):
            location_cluster = matched_location[clusterlabel == k]
            if location_cluster.shape[0] == 1:  # Only one point in cluster
                continue
    
            with Pool(5) as pool:  # Parallel processing with 5 cores
                results = pool.starmap(RASP.fx_1NN, [(i, location_cluster) for i in range(location_cluster.shape[0])])
            
            dist_val[count] = sum(results)
        
        dist_val = dist_val[~np.isnan(dist_val)]  # Remove any NaN values
        return np.sum(dist_val) / len(clusterlabel)


    @staticmethod
    def reconstruct_gene(adata, 
                         smoothed_pca_matrix, 
                         weights,
                         gene_name='test', 
                         quantile_prob=0.001,
                         scale = False,
                         threshold_method = 'ALRA',
                         rank_k = 20):

 
        """
        Restore true biological zeros while considering excess zeros and apply scaling.
        
        Parameters:
        - adata: AnnData object containing the gene expression data.
        - smoothed_pca_matrix: Spatially smoothed PC matrix.
        - weights: Weights from the initial PCA.
        - gene_name: The specific gene for which to reconstruct.
        - quantile_prob: The quantile threshold to use for determining biological zeros.
        - scale: Bool indicator to scale values to match original expression.
        - threshold_method: ALRA or Zero, how to deal with restoration of biological zeros to the imputed data. 
        - rank_k: number of PCs to use in the reconstruction. 
        
        Returns:
        - adata: Updated AnnData object with reconstructed gene expression in the adata.obs[restored_{gene_name}] slot.
        """
        
        # Get the original gene expression data
        original_data = adata.X
        indices = range(rank_k)
        gene_index = adata.var.index.get_loc(gene_name)
        original_expression = original_data[:, gene_index].toarray().flatten() if isinstance(original_data, csr_matrix) else original_data[:, gene_index]
    
        
    
            #subset to get rank k reconstruction: 
        indices = range(rank_k)
        smoothed_pca_matrix = smoothed_pca_matrix[:,indices]
    
        gene_weights = weights[indices, gene_index]
        reconstructed_gene_expression = np.dot(smoothed_pca_matrix, gene_weights)
    
        delta_mean = np.mean(original_expression)
        reconstructed_gene_expression += delta_mean
    
        # Calculate the quantile threshold using absolute value
        #note: the ALRA method uses the abs of the quantile and then restores the expression of some cell cells that are non-zero 
        # from the original expression matrix. This is different than what I am doing which is taking whatever is smaller: the threshold or 
        # zero. 
    
    
        if threshold_method == 'Zero':
            threshold_value = np.quantile(reconstructed_gene_expression, quantile_prob)
            threshold_value = max(0,threshold_value)
        
            print(f'Threshold read value: {np.quantile(reconstructed_gene_expression, quantile_prob)}')
            
            
            # Restore the biological zeros based on the excess zeros logic
            restored_expression = reconstructed_gene_expression.copy()
            print(f'Number of cells below the threshold: {np.sum(restored_expression < threshold_value)}')
            print(f'Number of cells below zero: {np.sum(restored_expression < 0)}')
        
            restored_expression[restored_expression < threshold_value] = 0
    
            
        
            #in case negative values remain, set those to zero as well! 
            #restored_expression[restored_expression < 0] = 0 
        
            print(f'Number of cells with zero before imputation:{np.sum(original_expression==0)}')
            print(f'Number of cells with zero AFTER imputation:{np.sum(restored_expression==0)}')
    
        if threshold_method == 'ALRA':
            threshold_value =  np.abs(np.quantile(reconstructed_gene_expression, quantile_prob))
            print(f'Threshold (absolute value for ALRA method): {threshold_value}')
            restored_expression = reconstructed_gene_expression.copy()
            print(f'Number of cells below the threshold: {np.sum(restored_expression < threshold_value)}')
            print(f'Number of cells below zero: {np.sum(restored_expression < 0)}')
            restored_expression[restored_expression < threshold_value] = 0
            
            # Restore original values for Non-Zero entries that were thresholded out
            mask_thresholded_to_zero = (reconstructed_gene_expression < threshold_value) & (original_expression > 0)
    
            #note: the ALRA method restors the original expression here. What I am doing is instead restoring the 
            #reconstructed expression, as long as it is not zero! 
            #restored_expression[mask_thresholded_to_zero] = original_expression[mask_thresholded_to_zero]
            restored_expression[mask_thresholded_to_zero] = reconstructed_gene_expression[mask_thresholded_to_zero]
            print(f'Number of cells restored to original values:{np.sum(mask_thresholded_to_zero != 0)}')
            print(f'Number of cells that where negative: {np.sum(reconstructed_gene_expression[mask_thresholded_to_zero]<0)}')
    
            #finally, set anything that is still negative to zero, should be a very small number of cells! 
            restored_expression[restored_expression < 0] = 0
            
        if scale:
            
    
            # Now, perform scaling based on the original and restored values
            sigma_1 = np.std(restored_expression[restored_expression > 0])
            sigma_2 = np.std(original_expression[original_expression > 0])
            mu_1 = np.mean(restored_expression[restored_expression > 0])
            mu_2 = np.mean(original_expression[original_expression > 0])
        
            # Avoid division by zero
            if sigma_1 == 0:
                sigma_1 = 1e-10  # Or choose to keep restored_expression intact
            
            # Determine scaling factors
            scaling_factor = sigma_2 / sigma_1
            offset = mu_2 - (mu_1 * scaling_factor)
        
            # Apply scaling
            restored_expression = restored_expression * scaling_factor + offset
        
            # If case scaling results in any negative values, turn those to zero as well! 
            #print(f'Number of cells turned negative after scaling: {np.sum(restored_expression_scaled < 0)}')
            restored_expression[restored_expression < 0] = 0
            
    
        # Store the final restored gene expression back into adata
        adata.obs['restored_' + gene_name] = restored_expression.flatten()
            
        return adata

    @staticmethod
        
    def calculate_local_density(coords, neighborhood_size):
        """
        Calculate the local density of each coordinate in a given neighborhood size.
    
        Parameters:
        - coords: ndarray of shape (n_samples, n_features) - spatial coordinates of cells.
        - neighborhood_size: float - the radius within which to calculate the local density.
    
        Returns:
        - local_density: ndarray of shape (n_samples,) - local density for each coordinate.
        """
        tree = KDTree(coords)
        
        # Query the tree to find the number of points within the neighborhood size (radius)
        densities = []
        for point in tqdm(coords):
            indices = tree.query_ball_point(point, r=neighborhood_size)
            # The density is the number of points within the neighborhood divided by the volume of the neighborhood
            density = len(indices) / (np.pi * neighborhood_size ** 2)  # Assuming 2D coordinates, adjust formula for 3D
            densities.append(density)
        
        return np.array(densities)
