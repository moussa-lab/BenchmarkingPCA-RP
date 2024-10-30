# Importing all the required libraries
import numpy as np
import pandas as pd
import shutil
import os
import time
import pickle
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from umap.umap_ import UMAP
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    mutual_info_score,
    accuracy_score
)
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
# Function for Spherical KMeans
class SphericalKMeans(KMeans):
    def __init__(self, n_clusters=2, max_iter=300, random_state=None):
        super(SphericalKMeans, self).__init__(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)

    def fit(self, X, y=None, sample_weight=None):
        X_normalized = normalize(X, norm='l2')
        return super(SphericalKMeans, self).fit(X_normalized, y=y, sample_weight=sample_weight)

    def predict(self, X, sample_weight=None):
        X_normalized = normalize(X, norm='l2')
        return super(SphericalKMeans, self).predict(X_normalized, sample_weight=sample_weight)

    def fit_predict(self, X, y=None, sample_weight=None):
        X_normalized = normalize(X, norm='l2')
        return super(SphericalKMeans, self).fit_predict(X_normalized, y=y, sample_weight=sample_weight)
# Function to calculate the dunn index
def dunn_index(X, labels):
    
    # Calculate pairwise cosine distances
    distances = squareform(pdist(X, metric='cosine'))
    unique_labels = np.unique(labels)

    intra_cluster_max = 0
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) > 1: 
            intra_distances = distances[np.ix_(indices, indices)]
            np.fill_diagonal(intra_distances, np.nan)  # Ignore self-distances by setting them to NaN
            max_intra = np.nanmax(intra_distances)
            intra_cluster_max = max(intra_cluster_max, max_intra)

    inter_cluster_min = np.inf
    for i, label_i in enumerate(unique_labels[:-1]):
        for label_j in unique_labels[i + 1:]:
            indices_i = np.where(labels == label_i)[0]
            indices_j = np.where(labels == label_j)[0]
            if len(indices_i) > 0 and len(indices_j) > 0: 
                inter_distances = distances[np.ix_(indices_i, indices_j)]
                min_inter = np.min(inter_distances)
                inter_cluster_min = min(inter_cluster_min, min_inter)
                
    if intra_cluster_max == 0:
        intra_cluster_max = np.nan

    return inter_cluster_min / intra_cluster_max if intra_cluster_max > 0 else 0
# Function for Gap Statistic
def gap_statistic(data, refs=None, n_refs=10, k_max=10):

    if refs is None:
        shape = data.shape
        tops = data.max(axis=0)
        bottoms = data.min(axis=0)
        dists = np.diag(tops - bottoms)
        rands = np.random.random_sample(size=(n_refs, shape[0], shape[1]))
        refs = rands.dot(dists) + bottoms

    gaps = np.zeros(k_max)
    for k in range(1, k_max + 1):
        # Fit to original data
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(data)
        Wk = km.inertia_

        # Compute Wk for reference data
        Wk_refs = np.zeros(n_refs)
        for i in range(n_refs):
            km_ref = KMeans(n_clusters=k, random_state=42)
            km_ref.fit(refs[i])
            Wk_refs[i] = km_ref.inertia_

        # Compute Gap statistic
        gaps[k-1] = np.log(np.mean(Wk_refs)) - np.log(Wk)

    return gaps
# Function to map predicted labels to true labels using Hungarian Algorithm
def calculate_accuracy(true_labels, predicted_labels):
    if true_labels is None:
        return None
    contingency_matrix = pd.crosstab(pd.Series(true_labels, name='True'), pd.Series(predicted_labels, name='Predicted'))
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix.values)
    total = contingency_matrix.values[row_ind, col_ind].sum()
    accuracy = total / np.sum(contingency_matrix.values)
    return accuracy

# Function to compute WCSS (Within-Cluster Sum of Squares)
def compute_wcss(data, labels):
    wcss = 0
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = data[labels == label]
        centroid = cluster_points.mean(axis=0)
        wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss
# Function to perform clustering using Spherical KMeans and evaluate metrics
def cluster_and_evaluate(data, true_labels, n_clusters, labeled):
    # Initialize Spherical KMeans
    spherical_kmeans = SphericalKMeans(n_clusters=n_clusters, random_state=42)
    predicted_labels = spherical_kmeans.fit_predict(data)
    
    metrics = {}
    if labeled:
        # Calculate accuracy and mutual information if labels are available
        accuracy = calculate_accuracy(true_labels, predicted_labels)
        mutual_info = mutual_info_score(true_labels, predicted_labels)
        metrics['Accuracy'] = accuracy
        metrics['Mutual Information'] = mutual_info
        print(f"Clustering Accuracy: {accuracy:.4f}, Mutual Information: {mutual_info:.4f}")
        # Calculate Dunn index and Adjusted Rand Index
        dunn = dunn_index(data, predicted_labels)
        ari = adjusted_rand_score(true_labels, predicted_labels) if true_labels is not None else None
        metrics['Dunn Index'] = dunn
        metrics['Adjusted Rand Index'] = ari
        print(f"Dunn Index: {dunn:.4f}, Adjusted Rand Index: {ari:.4f}")

    return predicted_labels, metrics
# Function to plot UMAP results
def plot_umap_results(results, labels, title, color_palette, folder="plots"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.figure(figsize=(10, 6))
    if labels is not None:
        scatter = plt.scatter(results[:, 0], results[:, 1], c=labels, cmap=color_palette, s=50, edgecolor='k')
        plt.colorbar(scatter, label='Label Category')
    else:
        scatter = plt.scatter(results[:, 0], results[:, 1], cmap=color_palette, s=50, edgecolor='k')
    plt.title(title, fontsize=16)
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{folder}/{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()
def plot_wcss_boxplot(results_df, dataset_name, metrics_folder, base_palette):
    methods = results_df['Reduction Method'].unique()
    clustering_methods = results_df['Clustering Method'].unique()
    
    # Check if WCSS is present in the results
    if 'WCSS' not in results_df.columns or results_df['WCSS'].isnull().all():
        print("WCSS metric is missing or has no valid data.")
        return

    # Define broader bins for the components
    def categorize_components(x):
        if 5 <= x < 25:
            return 'Components < 25'
        else: 
            return 'Components >= 25'

    # Apply categorization
    results_df['Component Category'] = results_df['Components'].apply(categorize_components)

    # Generate color palette for the component categories
    component_categories = results_df['Component Category'].unique()
    palette = sns.color_palette('husl', n_colors=len(component_categories))
    component_colors = dict(zip(component_categories, palette))
    
    # Plotting WCSS as a box plot for each clustering method
    for clust_method in clustering_methods:
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            x='Reduction Method',
            y='WCSS',
            hue='Component Category',
            data=results_df[results_df['Clustering Method'].str.strip() == clust_method],
            palette=component_colors,
            showfliers=False
        )
        plt.title(f'WCSS Box Plot for {clust_method} - {dataset_name}', fontsize=16)
        plt.xlabel('Dimensionality Reduction Method', fontsize=12)
        plt.ylabel('WCSS Value', fontsize=12)
        plt.xticks(rotation=45)
        
        # Position the legend inside the plot
        plt.legend(
            title='Component Category',
            loc='lower right',
            bbox_to_anchor=(1.0, 0.0),
            framealpha=0.5,
            borderaxespad=0.1
        )
        
        plt.tight_layout()
        plot_filename = f"WCSS_Boxplot_{clust_method}_{dataset_name}.png"
        plt.savefig(os.path.join(metrics_folder, plot_filename), bbox_inches='tight', dpi=300)
        plt.close()
# Function to plot clustering metrics and save plots
def plot_metrics_and_timing(results_df, timing_df, dataset_name, metrics_folder, timing_folder, metrics_palette, timing_palette, marker_style):
    methods = results_df['Reduction Method'].unique()
    clustering_methods = results_df['Clustering Method'].unique()
    
    # Determine if the dataset is labeled
    if results_df['Adjusted Rand Index'].notnull().any():
        labeled = True
    else:
        labeled = False

    # Define metrics based on dataset type
    if labeled:
        metrics = ['Adjusted Rand Index', 'WCSS', 'Mutual Information', 'Accuracy',
                'Dunn Index', 'Silhouette Score', 'Gap Statistic',
                'Calinski-Harabasz Index', 'Davies-Bouldin Index']
    else:
        metrics = ['Dunn Index', 'WCSS', 'Silhouette Score', 'Gap Statistic',
                'Calinski-Harabasz Index', 'Davies-Bouldin Index']
    
    # Generate color palettes for methods
    num_methods = len(methods)
    metrics_palette_colors = sns.color_palette(metrics_palette, n_colors=num_methods)
    metrics_method_colors = dict(zip(methods, metrics_palette_colors))
    # Use the custom timing palette provided
    timing_palette_colors = timing_palette
    timing_method_colors = dict(zip(methods, timing_palette_colors))
    
    # Plotting clustering indices 
    for clust_method in clustering_methods:
        for index in metrics:
            if index not in results_df.columns or results_df[index].isnull().all():
                continue
            plt.figure(figsize=(10, 5))
            for method in methods:
                method_data = results_df[(results_df['Reduction Method'].str.strip() == method) & 
                                        (results_df['Clustering Method'].str.strip() == clust_method)]
                if not method_data.empty:
                    sns.regplot(
                        x='Components', 
                        y=index, 
                        data=method_data, 
                        label=f'{method} Linear Trend', 
                        marker=marker_style, 
                        scatter_kws={'s': 50},
                        color=metrics_method_colors[method]  
                    )
            plt.title(f'{index} - Linear Trend for {clust_method} - {dataset_name}', fontsize=16)
            plt.xlabel('Number of Components', fontsize=12)
            plt.ylabel(f'{index} Value', fontsize=12)
            plt.legend()
            # Cap accuracy at 1
            if index == 'Accuracy':
                plt.ylim(0, 1)
            plt.tight_layout()
            plot_filename = f"{clust_method}_{index}_{dataset_name}_Linear.png"
            plt.savefig(os.path.join(metrics_folder, plot_filename), bbox_inches='tight', dpi=300)
            plt.close()
    
    # Plotting execution times
    plt.figure(figsize=(14, 6))
    for method in methods:
        method_data = timing_df[timing_df['Reduction Method'].str.strip() == method]
        if not method_data.empty:
            sns.regplot(
                x='Components', 
                y='Reduction Time', 
                data=method_data, 
                label=f'{method} Reduction Time', 
                marker=marker_style,  
                scatter_kws={'s': 50},
                color=timing_method_colors[method]
            )
    plt.xlabel('Number of Components', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title(f'Execution Time Comparison for {dataset_name}', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plot_filename = f"Execution_Times_{dataset_name}_Trend.png"
    plt.savefig(os.path.join(timing_folder, plot_filename), bbox_inches='tight', dpi=300)
    plt.close()
# Function to plot PCA with 2 Components
def plot_pca_results(data, labels, title, color_palette, folder="plots"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.figure(figsize=(10, 6))
    if labels is not None:
        scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=color_palette, s=50, edgecolor='k')
        plt.colorbar(scatter, label='Label Category')
    else:
        scatter = plt.scatter(data[:, 0], data[:, 1], cmap=color_palette, s=50, edgecolor='k')
    plt.title(title, fontsize=16)
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{folder}/{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()
# Function to plot Random Projection results with 2 Components
def plot_random_projection_results(data, labels, title, color_palette, projection_type, folder="plots"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.figure(figsize=(10, 6))
    if labels is not None:
        scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=color_palette, s=50, edgecolor='k')
        plt.colorbar(scatter, label='Label Category')
    else:
        scatter = plt.scatter(data[:, 0], data[:, 1], cmap=color_palette, s=50, edgecolor='k')
    plt.title(f'{title}', fontsize=16)
    plt.xlabel(f'{projection_type} Component 1', fontsize=12)
    plt.ylabel(f'{projection_type} Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{folder}/{projection_type}_{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()
from scipy.cluster.hierarchy import dendrogram, linkage

def plot_dendrogram(Z, **kwargs):
    # Plot the dendrogram based on the linkage matrix Z
    dendrogram(Z, **kwargs)
# Function to process a single dimensionality reduction and clustering method
def process_component(data, method, n_components, n_clusters, pca_type=None, true_labels=None):
    if method == 'PCA':
        if pca_type == 'full':
            reducer = PCA(n_components=n_components, svd_solver='full')
            method_label = 'PCA_full'
        elif pca_type == 'randomized':
            reducer = PCA(n_components=n_components, svd_solver='randomized')
            method_label = 'PCA_randomized'
    elif method == 'SparseRandomProjection':
        reducer = SparseRandomProjection(n_components=n_components, random_state=42)
        method_label = 'SparseRandomProjection'
    else:  
        reducer = GaussianRandomProjection(n_components=n_components, random_state=42)
        method_label = 'GaussianRandomProjection'
        
    # Perform dimensionality reduction    
    start_time = time.time()
    reduced_data = reducer.fit_transform(data)
    reduction_time = time.time() - start_time

    assert reduced_data.shape[0] == data.shape[0], "Mismatch between reduced data and original data!"
    
    norms = np.linalg.norm(reduced_data, axis=1)
    non_zero_indices = norms > 1e-10 
    num_zero_vectors = np.sum(~non_zero_indices)
    if num_zero_vectors > 0:
        print(f"[INFO] Found {num_zero_vectors} zero vectors in {method_label} with {n_components} components. Removing them before clustering.")
        reduced_data = reduced_data[non_zero_indices]
        if true_labels is not None:
            true_labels = true_labels[non_zero_indices].astype(int)

    if reduced_data.shape[0] < n_clusters:
        print(f"[WARNING] Number of samples after removing zero vectors ({reduced_data.shape[0]}) is less than the number of clusters ({n_clusters}). Skipping clustering for {method_label} with {n_components} components.")
        return [], []

    results = []
    timing_results = []

    # Clustering methods
    for clustering_method in ['Hierarchical', 'SphericalKMeans']:
        if clustering_method == 'Hierarchical':
            distance_matrix = pdist(reduced_data, metric='cosine')
            Z = linkage(distance_matrix, method='ward')
            predicted_labels = fcluster(Z, n_clusters, criterion='maxclust')
        else:  
            clusterer = SphericalKMeans(n_clusters=n_clusters, random_state=42)
            predicted_labels = clusterer.fit_predict(reduced_data)

        # Calculate WCSS
        wcss = compute_wcss(reduced_data, predicted_labels)

        # Calculate metrics
        if true_labels is not None:
            # Labeled Data Metrics
            accuracy = calculate_accuracy(true_labels, predicted_labels)
            mutual_info = mutual_info_score(true_labels, predicted_labels)
            adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
            gap = gap_statistic(reduced_data, n_refs=10, k_max=n_clusters)[n_clusters-1]  # Gap at k=n_clusters
            metrics = {
                'Mutual Information': mutual_info,
                'Accuracy': accuracy,
                'Dunn Index': None,
                'Gap Statistic': gap,
                'WCSS': wcss
            }
        else:
            # Unlabeled Data Metrics
            dunn = dunn_index(reduced_data, predicted_labels)
            gap = gap_statistic(reduced_data, n_refs=10, k_max=n_clusters)[n_clusters-1]  # Gap at k=n_clusters
            metrics = {
                'Mutual Information': None,
                'Accuracy': None,
                'Dunn Index': dunn,
                'Gap Statistic': gap,
                'WCSS': wcss
            }

        # Append results
        results.append({
            'Reduction Method': method_label,
            'Components': n_components,
            'Clustering Method': clustering_method,
            **metrics
        })

        # Append timing results
        timing_results.append({
            'Reduction Method': method_label,
            'Components': n_components,
            'Reduction Time': reduction_time
        })

    return results, timing_results
# Function to run experiments in parallel
def run_experiment(data, n_components_range, true_labels=None, n_clusters=2, n_jobs=64):
    tasks = [
        (method, n_components, pca_type, n_clusters)
        for method in ['PCA', 'SparseRandomProjection', 'GaussianRandomProjection']
        for n_components in n_components_range
        for pca_type in (['full', 'randomized'] if method == 'PCA' else [None])
    ]

    all_results = []
    all_timing_results = []

    with tqdm(total=len(tasks), desc="Processing Tasks") as pbar:
        # Parallel processing using joblib
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_component)(data, method, n_components, n_clusters, pca_type, true_labels)
            for method, n_components, pca_type, n_clusters in tasks
        )
        for result in results:
            all_results.extend(result[0])
            all_timing_results.extend(result[1])
            pbar.update(1)

    return all_results, all_timing_results
# Datasets and labels
datasets_info = {
    '50-50_Mixture': {
        'data_path': '../Datasets/Jurkat_Cleaned/Jurkat293T_Clean.csv',
        'label_path': '../Datasets/Jurkat_Cleaned/Jurkat293T_Clean_TrueLabels.csv',
        'color_palette': 'viridis',
        'type': 'labeled',
        'n_clusters': 2
     },
    'Labeled_PBMC': {
        'data_path': '../Datasets/PBMC-Zheng2017/PBMC_SC1.csv',
        'label_path': '../Datasets/PBMC-Zheng2017/PBMCLabels_SC1ClusterLabels.csv',
        'color_palette': 'plasma',
        'type': 'labeled',
        'n_clusters': 7
    },
    'Unlabeled_PBMC': {  # Unlabeled
        'data_path': '../Datasets/Unlabeled_PBMC/unlabled_PBMC.csv',
        'label_path': None,
        'color_palette': 'cividis',
        'type': 'unlabeled',
        'n_clusters': 6
    },
    'Covid19': {  # Unlabeled
        'data_path': '../Datasets/Covid19TCells/COVID19DataSC1.csv',
        'label_path': None,
        'color_palette': 'magma',
        'type': 'unlabeled',
        'n_clusters': 6
    }
}
def process_all_datasets(datasets_info, n_components_range, output_dir="output"):
    
    # Palettes for each section
    clustering_metrics_palette = 'magma'
    timing_palette = ['blue', 'orange', 'red', 'green']
    umap_palette = 'viridis'
    
    # Marker styles for datasets
    dataset_marker_styles = ['o', 's', 'D', '*', 'v', '<', '>', 'p', '^']
    dataset_markers = {}
    for dataset_name, marker in zip(datasets_info.keys(), dataset_marker_styles):
        dataset_markers[dataset_name] = marker

    if os.path.exists(output_dir):
        print(f"Clearing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created fresh output directory: {output_dir}")

    for dataset_name, info in datasets_info.items():
        print(f"\n=== Processing Dataset: {dataset_name} ===")
        # Create a unique output subfolder for each dataset
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        print(f"Created directory: {dataset_output_dir}")

        # Create subfolders for different plots
        umap_plots_folder = os.path.join(dataset_output_dir, "UMAP_Plots")
        pca_plots_folder = os.path.join(dataset_output_dir, "PCA_Plots")
        rp_plots_folder = os.path.join(dataset_output_dir, "RandomProjection_Plots")
        metrics_plots_folder = os.path.join(dataset_output_dir, "Metrics_Plots")
        timing_plots_folder = os.path.join(dataset_output_dir, "Timing_Plots")

        for folder in [umap_plots_folder, pca_plots_folder, rp_plots_folder, metrics_plots_folder, timing_plots_folder]:
            os.makedirs(folder, exist_ok=True)

        # Load dataset
        try:
            dataset = pd.read_csv(info['data_path'], index_col=0)
            print(f"Data Loaded Successfully. Shape: {dataset.shape}")
        except FileNotFoundError:
            print(f"Error: Data file not found for {dataset_name} at {info['data_path']}")
            continue
        except Exception as e:
            print(f"Error loading data for {dataset_name}: {e}")
            continue

        # Transpose data to have samples as rows and features as columns
        data_transposed = dataset.T.values  # Shape: (samples, features)

        if info['label_path'] is not None:
            try:
                labels_df = pd.read_csv(info['label_path'])
                # Reindex labels based on dataset.columns to align labels with samples
                labels_matched = labels_df.set_index('Unnamed: 0').reindex(dataset.columns)['x'].astype(int).values
                # Check for any missing labels after reindexing
                if np.isnan(labels_matched).any():
                    print(f"Warning: Some labels are missing for {dataset_name}. Dropping these samples.")
                    valid_indices = ~np.isnan(labels_matched)
                    data_transposed = data_transposed[valid_indices]
                    labels_matched = labels_matched[valid_indices].astype(int)
                print(f"Labels Loaded Successfully. Shape: {labels_matched.shape}")
            except FileNotFoundError:
                print(f"Error: Label file not found for {dataset_name} at {info['label_path']}")
                labels_matched = None
            except KeyError:
                print(f"Error: 'x' column not found in label file for {dataset_name}.")
                labels_matched = None
            except ValueError as ve:
                print(f"Error converting labels to integers for {dataset_name}: {ve}")
                labels_matched = None
            except Exception as e:
                print(f"Unexpected error loading labels for {dataset_name}: {e}")
                labels_matched = None
        else:
            labels_matched = None

        # Run experiment
        print(f"Running dimensionality reduction and clustering for {dataset_name}...")
        try:
            results, timing = run_experiment(
                data_transposed, 
                n_components_range, 
                true_labels=labels_matched, 
                n_clusters=info['n_clusters']
            )
        except Exception as e:
            print(f"Error during experiment for {dataset_name}: {e}")
            continue

        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        timing_df = pd.DataFrame(timing)

        # Save results
        results_save_path = os.path.join(dataset_output_dir, f"{dataset_name}_results.csv")
        timing_save_path = os.path.join(dataset_output_dir, f"{dataset_name}_timing.csv")
        try:
            results_df.to_csv(results_save_path, index=False)
            timing_df.to_csv(timing_save_path, index=False)
            print(f"Results saved to {results_save_path} and {timing_save_path}")
        except Exception as e:
            print(f"Error saving results for {dataset_name}: {e}")
        # Plot WCSS boxplot for this dataset
        print("Plotting WCSS boxplot for this dataset...")
        try:
            plot_wcss_boxplot(
                results_df,
                dataset_name,
                metrics_plots_folder,
                info['color_palette']
            )
            print(f"WCSS boxplot saved in {metrics_plots_folder}")
        except Exception as e:
            print(f"Error plotting WCSS boxplot for {dataset_name}: {e}")
            
        # Plot clustering metrics and timing for this dataset
        print("Plotting clustering metrics and execution times for this dataset...")
        try:
            plot_metrics_and_timing(
                results_df, 
                timing_df, 
                dataset_name, 
                metrics_plots_folder, 
                timing_plots_folder, 
                metrics_palette=clustering_metrics_palette,
                timing_palette=timing_palette,
                marker_style=dataset_markers[dataset_name]
            )
            print(f"Clustering metrics and execution times plots saved in {metrics_plots_folder} and {timing_plots_folder}")
        except Exception as e:
            print(f"Error plotting metrics and timing for {dataset_name}: {e}")
                
        # Store evaluation results in a dictionary for each dataset
        evaluation_results = {
            'Method': [],
            'Accuracy': [],
            'Mutual Information': [],
            'Dunn Index': [],
            'Adjusted Rand Index': []
        }

        # Visualization with UMAP, PCA, and Random Projections
        print("Generating visualizations (UMAP, PCA, Random Projections)...")
        if info['type'] == 'labeled':
            try:
                
                pca_full = PCA(n_components=500, svd_solver='full')
                pca_full_result = pca_full.fit_transform(data_transposed)
                pca_randomized = PCA(n_components=500, svd_solver='randomized', random_state=42)
                pca_randomized_result = pca_randomized.fit_transform(data_transposed)
                grp = GaussianRandomProjection(n_components=500, random_state=42)
                grp_result = grp.fit_transform(data_transposed)
                srp = SparseRandomProjection(n_components=500, random_state=42)
                srp_result = srp.fit_transform(data_transposed)

                # Apply UMAP for visualization
                umap_model = UMAP(n_neighbors=5, min_dist=0.3, n_components=3, random_state=42) # `n_components=3` for 3D UMAP
                pca_full_umap = umap_model.fit_transform(pca_full_result)
                pca_randomized_umap = umap_model.fit_transform(pca_randomized_result)  
                grp_umap = umap_model.fit_transform(grp_result)
                srp_umap = umap_model.fit_transform(srp_result)
                
                # Perform clustering and evaluation
                for method_name, umap_data in zip(
                    ['PCA Full', 'PCA Randomized', 'GRP', 'SRP'],
                    [pca_full_umap, pca_randomized_umap, grp_umap, srp_umap]
                ):
                    print(f"Evaluating {method_name}... for {dataset_name}")
                    predicted_labels, metrics = cluster_and_evaluate(
                        umap_data, labels_matched, info['n_clusters'], info['type'] == 'labeled'
                    )
                    # Collect results for saving
                    evaluation_results['Method'].append(method_name)
                    evaluation_results['Accuracy'].append(metrics.get('Accuracy'))
                    evaluation_results['Mutual Information'].append(metrics.get('Mutual Information'))
                    evaluation_results['Dunn Index'].append(metrics.get('Dunn Index'))
                    evaluation_results['Adjusted Rand Index'].append(metrics.get('Adjusted Rand Index'))
                    
                # Convert evaluation results to DataFrame
                evaluation_df = pd.DataFrame(evaluation_results)

                # Save the evaluation results as a CSV file
                evaluation_save_path = os.path.join(dataset_output_dir, f"{dataset_name}_evaluation.csv")
                evaluation_df.to_csv(evaluation_save_path, index=False)
                print(f"Evaluation results saved to {evaluation_save_path}")

                # Plot UMAP results
                plot_umap_results(
                    pca_full_umap, 
                    labels_matched, 
                    f'PCA Full Locality Preservation - {dataset_name}', 
                    umap_palette, 
                    folder=umap_plots_folder
                )
                
                # Plot UMAP results for Randomized PCA
                plot_umap_results(
                    pca_randomized_umap, 
                    labels_matched, 
                    f'PCA Randomized Locality Preservation - {dataset_name}', 
                    umap_palette, 
                    folder=umap_plots_folder
                )
                
                plot_umap_results(
                    grp_umap, 
                    labels_matched, 
                    f'GRP Locality Preservation - {dataset_name}', 
                    umap_palette, 
                    folder=umap_plots_folder
                )
                plot_umap_results(
                    srp_umap, 
                    labels_matched, 
                    f'SRP Locality Preservation - {dataset_name}', 
                    umap_palette, 
                    folder=umap_plots_folder
                )
                print(f"UMAP plots saved in {umap_plots_folder}")
                

                # Plot PCA with 2 Components
                pca_2d = PCA(n_components=2, random_state=42).fit_transform(data_transposed)
                plot_pca_results(
                    pca_2d, 
                    labels_matched, 
                    f'PCA 2D - {dataset_name}', 
                    umap_palette, 
                    folder=pca_plots_folder
                )
                print(f"PCA plots saved in {pca_plots_folder}")

                # Plot Gaussian Random Projection with 2 Components
                grp_2d = GaussianRandomProjection(n_components=2, random_state=42).fit_transform(data_transposed)
                plot_random_projection_results(
                    grp_2d, 
                    labels_matched, 
                    f'GRP 2D - {dataset_name}', 
                    umap_palette, 
                    'GRP',
                    folder=rp_plots_folder
                )
                print(f"Gaussian Random Projection plots saved in {rp_plots_folder}")

                # Plot Sparse Random Projection with 2 Components
                srp_2d = SparseRandomProjection(n_components=2, random_state=42).fit_transform(data_transposed)
                plot_random_projection_results(
                    srp_2d, 
                    labels_matched, 
                    f'SRP 2D - {dataset_name}', 
                    umap_palette, 
                    'SRP',
                    folder=rp_plots_folder
                )
                print(f"Sparse Random Projection plots saved in {rp_plots_folder}")
            except Exception as e:
                print(f"Error during visualization for {dataset_name}: {e}")
        else:
            print(f"Skipping visualization for {dataset_name} as it is an unlabeled dataset.")

        print(f"=== Finished Processing Dataset: {dataset_name} ===\n")
def run_pca_experiment(data_path, components_list, output_dir, dataset_name):
    # Load the dataset
    try:
        dataset = pd.read_csv(data_path, index_col=0)
        print(f"Data Loaded Successfully for {dataset_name}. Shape: {dataset.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    except Exception as e:
        print(f"Error loading data for {dataset_name}: {e}")
        return

    data_transposed = dataset.T.values

    # Dictionary to store column sums and PCA components
    column_sums = {}
    pca_components = {}

    for n_components in components_list:
        print(f"Running PCA with {n_components} components for {dataset_name}...")
        try:
            # Initialize PCA with full SVD solver
            pca = PCA(n_components=n_components, svd_solver='full')
            pca_result = pca.fit_transform(data_transposed)

            # Calculate column sum
            col_sum = pca_result.sum(axis=0)
            column_sums[n_components] = col_sum
            pca_components[n_components] = pca_result

            print(f"Column sums for {n_components} components: {col_sum}")
        except Exception as e:
            print(f"Error during PCA for {n_components} components in {dataset_name}: {e}")
            continue

    # Save the column sums to a CSV file
    if column_sums:
        column_sums_df = pd.DataFrame.from_dict(column_sums, orient='index')
        column_sums_df.index.name = 'Components'
        output_path_sum = os.path.join(output_dir, f"{dataset_name}_column_sums.csv")
        column_sums_df.to_csv(output_path_sum)
        print(f"Column sums saved to {output_path_sum}")

    # Save each PCA component set in a separate sheet in an Excel file
    if pca_components:
        pca_output_path = os.path.join(output_dir, f"{dataset_name}_pca_components.xlsx")
        with pd.ExcelWriter(pca_output_path) as writer:
            for n_components, pca_result in pca_components.items():
                pca_df = pd.DataFrame(pca_result)
                sheet_name = f"PCA_{n_components}"
                pca_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"PCA components saved to {pca_output_path}")
        
def load_and_plot_and_save_combined_column_sums(file_path1, dataset_name1, file_path2, dataset_name2, save_path):
    try:
        # Load the column sums CSV files
        column_sums_df1 = pd.read_csv(file_path1, index_col='Components')
        column_sums_df2 = pd.read_csv(file_path2, index_col='Components')
        print(f"Column sums data loaded successfully for {dataset_name1} and {dataset_name2}.")

        # Filter the specific components we are interested in
        components_to_plot = [200, 400, 600, 800, 1000]
        filtered_df1 = column_sums_df1.loc[components_to_plot]
        filtered_df2 = column_sums_df2.loc[components_to_plot]

        # Plotting
        plt.figure(figsize=(10, 6))
        
        # Mean across all PCA components for each set of components for both datasets
        total_sums1 = filtered_df1.mean(axis=1)
        total_sums2 = filtered_df2.mean(axis=1)
        
        plt.plot(filtered_df1.index, total_sums1, marker='o', label=f'Total Column Mean - {dataset_name1}')
        plt.plot(filtered_df2.index, total_sums2, marker='s', linestyle='--', label=f'Total Column Mean - {dataset_name2}')

        plt.title('Column Mean vs. Number of Components for Both Datasets', fontsize=16)
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Total Column Mean', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Plot saved successfully at {save_path}.")
    except FileNotFoundError:
        print(f"Error: Column sums file not found for one of the datasets.")
    except Exception as e:
        print(f"Error loading or plotting data: {e}")

# File paths for the column sums CSV files
labeled_pbmc_file = './pca_results/Labeled_PBMC_column_sums.csv'
unlabeled_pbmc_file = './pca_results/Unlabeled_PBMC_column_sums.csv'
save_plot_path = './pca_results/Combined_Column_Means_Plot.png'

# Plotting and saving for both datasets
load_and_plot_and_save_combined_column_sums(labeled_pbmc_file, 'Labeled PBMC', unlabeled_pbmc_file, 'Unlabeled PBMC', save_plot_path)

if __name__ == "__main__":
    # Paths to the PBMC datasets
    pbmc_data_paths = {
        'Labeled_PBMC': '../Datasets/PBMC-Zheng2017/PBMC_SC1.csv',
        'Unlabeled_PBMC': '../Datasets/Unlabeled_PBMC/unlabeled_PBMC.csv'  # fixed typo here
    }

    # Components to investigate
    components_to_test = [200, 400, 600, 800, 1000]

    # Output directory for saving results
    output_directory = "./pca_results"
    os.makedirs(output_directory, exist_ok=True)

    # Run the PCA experiment for each dataset
    for dataset_name, data_path in pbmc_data_paths.items():
        print(f"\n=== Running PCA Experiment for {dataset_name} ===")
        run_pca_experiment(data_path, components_to_test, output_directory, dataset_name)

    # File paths for the column sums CSV files
    labeled_pbmc_file = './pca_results/Labeled_PBMC_column_sums.csv'
    unlabeled_pbmc_file = './pca_results/Unlabeled_PBMC_column_sums.csv'
    save_plot_path = './pca_results/Combined_Column_Means_Plot.png'

    # Plotting and saving for both datasets
    load_and_plot_and_save_combined_column_sums(
        labeled_pbmc_file,
        'Labeled PBMC',
        unlabeled_pbmc_file,
        'Unlabeled PBMC',
        save_plot_path
    )

    # Range for number of components
    n_components_range = list(range(5, 25, 1)) + list(range(25, 1001, 25))

    # Output directory for full experiment
    output_directory = "./output"

    # Process all datasets
    process_all_datasets(datasets_info, n_components_range, output_dir=output_directory)
    
        # PBMC datasets
    pbmc_data_paths = {
        'Labeled_PBMC': '../Datasets/PBMC-Zheng2017/PBMC_SC1.csv',
        'Unlabeled_PBMC': '../Datasets/Unlabeled_PBMC/unlabled_PBMC.csv'
    }

    # Components to investigate
    components_to_test = [200, 400, 600, 800, 1000]

    # Output directory for saving results
    output_directory = "./pca_results"
    os.makedirs(output_directory, exist_ok=True)

    # Run the experiment for each dataset
    for dataset_name, data_path in pbmc_data_paths.items():
        print(f"\n=== Running PCA Experiment for {dataset_name} ===")
        run_pca_experiment(data_path, components_to_test, output_directory, dataset_name)
