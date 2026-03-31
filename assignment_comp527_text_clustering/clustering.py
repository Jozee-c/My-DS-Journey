import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from preprocessing import load_data, preprocess_text, vectorize, reduce_and_normalize


# Configuration.
K_MIN = 1        # silhouette plot range start (score undefined at K=1)
K_MAX = 10       # silhouette plot range end
OPTIMAL_K = 4    # chosen K for clustering (based on silhouette analysis and manual inspection)
N_INIT = 10      # number of KMeans initialisations


# 1. Silhouette analysis.
def silhouette_analysis(X, k_min=K_MIN, k_max=K_MAX):
    '''
    Silhouette analysis using k from k_min to k_max.
    '''
    scores = {}
    k_values = list(range(k_min, k_max + 1))
    # Iterate over k values, perform KMeans clustering, and compute silhouette scores.
    for k in k_values:
        if k == 1:
            # Silhouette is undefined for K=1
            scores[k] = np.nan
            continue

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=N_INIT, max_iter=300)
        labels = kmeans.fit_predict(X)

        score = silhouette_score(X, labels)
        scores[k] = score
        print(f'k={k}, Silhouette score = {score:.4f}')

    return scores


# 2. Plot silhouette scores for different k values.
def plot_silhouette_scores(scores, output_path='silhouette_analysis_plot.png'):
    '''Plot silhouette scores for different k values.'''


    ks = [k for k in sorted(scores.keys()) if not np.isnan(scores[k])]
    vals = [scores[k] for k in ks]

    # ks = sorted(scores.keys())
    # vals = [scores[k] if not np.isnan(scores[k]) else 0.0 for k in ks]

    plt.figure(figsize=(8, 6))
    plt.plot(ks, vals, marker='o', linewidth=2, color='skyblue')
    plt.xlabel('k')
    plt.ylabel('silhouette_score')
    plt.title('Silhouette analysis')
    plt.xticks(ks)

    # Save the silhouette coefficient plot.
    plt.savefig(output_path, dpi=150)

    print(f'silhouette_analysis_plot saved → {output_path}')


# 3. Clustering using the optimal k.
def clustering(X, k):
    """
    Clustering texts using the optimal k.
    Optimal k is determined based on silhouette analysis and manual inspection of clusters.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=N_INIT, max_iter=300)
    labels = kmeans.fit_predict(X)
    return labels


# 4. Save labels.
def save_labels(labels, output_path='labels.txt'):
    """Save cluster labels to a text file."""
    with open(output_path, 'w') as f:
        for label in labels:
            f.write(str(label + 1) + '\n')
    print(f'Labels have been saved to {output_path}')


if __name__ == '__main__':

    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data_train.txt'

    print(f'\n{'='*55}')
    print(' CA2 Clustering Pipeline')
    print(f' Input file : {data_path}')
    print(f'{'='*55}\n')

    print('Data loading...')
    texts = load_data(data_path)

    print('\nPreprocessing the text data...')
    cleaned_texts = preprocess_text(texts)

    print('\nTF-IDF vectorizing...')
    X = vectorize(cleaned_texts, max_features=3000)

    print('\nReducing dimensions and normalising...')
    X_normalized = reduce_and_normalize(X, n_components=100)

    print(f'\nSilhouette_analysing K={K_MIN} to {K_MAX}...')
    scores = silhouette_analysis(X_normalized)

    print('\nGenerating silhouette plot ...')
    plot_silhouette_scores(scores, output_path='silhouette_analysis_plot.png')

    print(f'\nClustering using optimal k ({OPTIMAL_K})...')
    labels = clustering(X_normalized, k=OPTIMAL_K)

    print('\nSaving labels...')
    save_labels(labels, output_path='label.txt')

    # Print a quick summary
    unique, counts = np.unique(labels, return_counts=True)
    print(f'\n{'─'*40}')
    print(f'  Cluster summary  (K={OPTIMAL_K})')
    print(f'{'─'*40}')
    for u, c in zip(unique, counts):
        print(f'  Cluster {u + 1}: {c:>5d} instances')
    print(f'{'─'*40}')
    # print(f'  Total instances in label.txt: {sum(valid_mask)} (valid) '
    #       f'+ {valid_mask.count(False)} (filtered)')
    print('\nDone.\n')
