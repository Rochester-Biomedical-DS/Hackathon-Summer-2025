import sys
import numpy as np
import pandas as pd

def train_with_pseudoinverse(csv_file):
    print("Training gene expression model using pseudoinverse...")
    # Load data
    data = pd.read_csv(csv_file, index_col=0)
    
    # Extract gene names and experiment names
    gene_names = data.index.tolist()
    experiment_names = data.columns.tolist()
    
    # Create feature matrix
    def parse_experiment_name(exp_name):
        parts = exp_name.split('+')
        perturbed_genes = []
        for part in parts:
            if part.startswith('g') and part[1:5].isdigit():
                gene_id = part.split('.')[0]
                perturbed_genes.append(gene_id)
        return perturbed_genes
    
    X = np.zeros((len(experiment_names), len(gene_names)))
    
    for exp_idx, exp_name in enumerate(experiment_names):
        perturbed_genes = parse_experiment_name(exp_name)
        for gene in perturbed_genes:
            if gene in gene_names:
                gene_idx = gene_names.index(gene)
                X[exp_idx, gene_idx] = 1
    
    # Add bias term
    X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    
    # Target matrix
    Y = data.values.T
    
    # Use pseudoinverse to avoid singular matrix issues
    coefficients = np.linalg.pinv(X_with_bias) @ Y
    print("Training complete.")
    
    return coefficients, gene_names

def predict_with_pseudoinverse(coefficients, gene_names, genes_to_perturb):
    # Create feature vector
    features = np.zeros(len(gene_names))
    for gene in genes_to_perturb:
        if gene in gene_names:
            gene_idx = gene_names.index(gene)
            features[gene_idx] = 1
    
    # Add bias term
    features_with_bias = np.append(features, 1.1)
    
    # Predict all gene expressions
    predictions = features_with_bias @ coefficients
    
    # Return as dictionary
    return {gene: pred for gene, pred in zip(gene_names, predictions)}

# Usage
def run_pseudo(gene1, gene2):
    gene_to_use = sys.argv[1], sys.argv[2]
    coefficients, gene_names = train_with_pseudoinverse('data/train_set.csv')
    genes_to_perturb = [gene_to_use[0], gene_to_use[1]]
    predictions = predict_with_pseudoinverse(coefficients, gene_names, genes_to_perturb)
    return predictions
