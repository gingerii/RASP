# Randomized Spatial PCA (RASP)

![RASP overview](figures/figure_1.png)


## Description

Here, we present Randomized Spatial PCA (RASP), a novel spatially aware dimensionality reduction method for spatial transcriptomics (ST) data. 
RASP is designed to be orders-of-magnitude faster than existing techniques, scale to ST data with hundreds of thousands of locations, support the 
flexible integration of non-transcriptomic covariates, and enable the reconstruction of de-noised and spatially smoothed expression values for individual genes. 
To achieve these goals, RASP uses a randomized two-stage principal component analysis (PCA) framework that leverages sparse matrix operations and configurable spatial smoothing.

## Features

- **High-Speed Performance**: RASP is optimized for fast processing of large spatial transcriptomics datasets.
- **Flexible Integration**: Seamlessly integrates non-transcriptomic covariates into the analysis.
- **Spatially Smoothed Values**: Produces reconstructed expression values that account for spatial context.
- **User-Friendly**: Designed to be accessible for researchers in spatial biology.

## Requirements

Dependencies are declared in `pyproject.toml` and installed automatically. The
manuscript results were produced with the following pinned versions (also in
`requirements.txt`):
```
- numpy==1.26.4
- pandas==2.2.2
- scanpy==1.10.1
- squidpy==1.2.2
- matplotlib==3.8.4
- scipy==1.13.1
- scikit-learn==1.5.0
- python-igraph==0.11.5
```
The `mclust` clustering option additionally requires `rpy2==3.5.16` and an R
installation with the `mclust` package; all other clustering methods
(`louvain`, `leiden`, `KMeans`, `walktrap`) are pure Python.

## Installation
Navigate to where you would like to install the package, clone the repo, and
install with pip:
```bash
git clone https://github.com/gingerii/RASP.git
cd RASP
pip install .
```
To enable the optional R-backed `mclust` clustering:
```bash
pip install ".[mclust]"   # then, in R: install.packages("mclust")
```
## Usage
See the tutorials folder for an end-to-end example. In brief:

```python
from rasp import RASP
import scanpy as sc

# adata: normalized expression in adata.X, coordinates in adata.obsm['spatial']
RASP.reduce(adata, n_pcs=20, n_neighbors=6, beta=2, platform='visium')
sc.pp.neighbors(adata, use_rep=adata.uns['RASP']['embedding_key'])
RASP.clustering(adata, n_clusters=7, method='leiden')
```

### Covariate integration (two-stage RASP)
RASP can integrate non-transcriptomic covariates (e.g. morphology or histology
features). When `covariates` are supplied, the smoothed stage-1 PC scores are
concatenated with the (optionally spatially smoothed) covariates and a second
randomized PCA produces the final embedding:

```python
# covariates: an (n_obs, d) array, adata.obs column name(s), or an adata.obsm key
RASP.reduce(adata, n_pcs=20, covariates=['morph_area', 'morph_ecc'])
# the integrated embedding is written to adata.obsm['X_pca_cov']; the key to use
# downstream is always adata.uns['RASP']['embedding_key']
sc.pp.neighbors(adata, use_rep=adata.uns['RASP']['embedding_key'])
```

Use `smooth_covariates` (bool or per-covariate list) and `scale_covariates` to
control covariate smoothing and z-scoring.

## Citation
If you use RASP in your research, please cite the following preprint: https://www.biorxiv.org/content/10.1101/2024.12.20.629785v1
