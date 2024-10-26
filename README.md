# A Benchmarking Study of Random Projections and Principal Components for Dimensionality Reduction Strategies in Single Cell Analysis

This repository contains Python scripts for evaluating various dimensionality reduction methods, including PCA (Principal Component Analysis) with Full SVD (Singular Value Decomposition) and Randomized SVD, Random Projection (Gaussian and Sparse), and UMAP, combined with clustering algorithms (Hierarchical clustering and Spherical KMeans) on scRNA-seq datasets. The code applies these techniques on different datasets, calculates clustering metrics, and generates visualizations.

## Table of Contents
- [Features](#features)
- [Datasets](#datasets)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Features

- **Dimensionality Reduction Methods**: Includes PCA, Gaussian Random Projection, Sparse Random Projection, and UMAP.
- **Clustering Algorithms**: Supports Spherical KMeans and Agglomerative (Hierarchical) clustering.
- **Metrics**: Computes clustering metrics including Dunn Index, Adjusted Rand Index, Mutual Information, Accuracy, Gap Statistic, and WCSS.
- **Visualization**: Generates 2D and 3D visualizations for UMAP, PCA, and Random Projections.
- **Parallel Processing**: Utilizes parallel processing to handle computationally heavy tasks efficiently.

## Datasets

The following datasets are used in this analysis:
- `50-50_Mixture`
- `Labeled_PBMC` and `Unlabeled_PBMC`
- `Covid19`

Each dataset should be placed in the `Datasets` folder with the following structure:

```
Datasets/
├── PBMC-Zheng2017/
│   ├── PBMC_SC1.csv
│   └── PBMCLabels_SC1ClusterLabels.csv
├── Jurkat_Cleaned/
│   ├── Jurkat293T_Clean.csv
│   └── Jurkat293T_Clean_TrueLabels.csv
├── Unlabeled_PBMC/
│   └── unlabeled_PBMC.csv
└── Covid19TCells/
    ├── COVID19DataSC1.csv
```

## Setup and Installation

### Prerequisites

- Python 3.7 or above
- Virtualenv (optional but recommended for creating a virtual environment)

### Installation

1. Clone the repository:

git clone https://github.com/moussa-lab/BenchmarkingPCA-RP.git

2. Create and activate a virtual environment:

3. Install the required packages:

pip install -r requirements.txt

## Usage

1. Run the main script for dimensionality reduction and clustering:

python main.py

## Results

Results are saved in `output` and `pca_results` directories:
- `UMAP_Plots/`: UMAP visualizations for different reduction methods.
- `PCA_Plots/` and `RandomProjection_Plots/`: 2D visualizations for PCA and Random Projection.
- `Metrics_Plots/` and `Timing_Plots/`: Evaluation metrics and timing results for each dataset.
- `Combined_Column_Means_Plot.png`: Mean plot for total column means across components in `pca_results`.

## Acknowledgments
This work is supported by the following grants: NSF-2341725, NIH-NCI K25CA270079, OU-BIC2.0

