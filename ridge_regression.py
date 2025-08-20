import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def train_gene_expression_model(csv_file, alpha=1.0):
    # Load data
    data = pd.read_csv(csv_file, index_col=0)
    
    # Extract gene names and experiment names
    gene_names = data.index.tolist()
    experiment_names = data.columns.tolist()
    
    print(f"Data shape: {data.shape}")
    print(f"Number of genes: {len(gene_names)}")
    print(f"Number of experiments: {len(experiment_names)}")
    
    # Parse experiment names to create feature matrix
    def parse_experiment_name(exp_name):
        parts = exp_name.split('+')
        perturbed_genes = []
        for part in parts:
            # Extract gene names (adjust this pattern based on your actual naming convention)
            if part.startswith('g') and part[1:5].isdigit():  # Adjust based on your gene naming
                gene_id = part.split('.')[0]  # Remove any trailing numbers
                perturbed_genes.append(gene_id)
        return perturbed_genes
    
    # Create feature matrix (which genes are perturbed in each experiment)
    X = np.zeros((len(experiment_names), len(gene_names)))
    
    for exp_idx, exp_name in enumerate(experiment_names):
        perturbed_genes = parse_experiment_name(exp_name)
        
        # Mark which genes are perturbed in this experiment
        for gene in perturbed_genes:
            if gene in gene_names:
                gene_idx = gene_names.index(gene)
                X[exp_idx, gene_idx] = 1
    
    # Target matrix (expression values for all genes in each experiment)
    Y = data.values.T  # Transpose to match experiments as rows
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target matrix shape: {Y.shape}")
    
    # Standardize features and targets
    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    Y_scaled = Y_scaler.fit_transform(Y)
    
    # Train Ridge regression models for each gene
    models = []
    for i in range(Y_scaled.shape[1]):
        model = Ridge(alpha=alpha)
        model.fit(X_scaled, Y_scaled[:, i])
        models.append(model)
    
    return models, X_scaler, Y_scaler, gene_names

def predict_expression(models, X_scaler, Y_scaler, gene_names, genes_to_perturb):
    """
    Predict gene expression when specific genes are perturbed
    """
    # Create feature vector
    features = np.zeros(len(gene_names))
    for gene in genes_to_perturb:
        if gene in gene_names:
            gene_idx = gene_names.index(gene)
            features[gene_idx] = 1
    
    # Scale features
    features_scaled = X_scaler.transform(features.reshape(1, -1))
    
    # Predict each gene's expression
    predictions_scaled = np.zeros(len(gene_names))
    for i, model in enumerate(models):
        predictions_scaled[i] = model.predict(features_scaled)
    
    # Inverse transform to get original scale
    predictions = Y_scaler.inverse_transform(predictions_scaled.reshape(1, -1))
    
    # Return as dictionary
    return {gene: pred for gene, pred in zip(gene_names, predictions[0])}

# Usage
try:
    models, X_scaler, Y_scaler, gene_names = train_gene_expression_model('data/train_set.csv', alpha=10.0)
    
    # Make predictions
    genes_to_perturb = ['g0037', 'g0083']
    predictions = predict_expression(models, X_scaler, Y_scaler, gene_names, genes_to_perturb)
    
    print(f"\nPredicted expression when perturbing {genes_to_perturb}:")
    for gene, expr in list(predictions.items())[:10]:
        print(f"{gene}: {expr:.6f}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()