import numpy as np
import pandas as pd


data = pd.read_csv('data/train_set.csv', index_col=0)
def parse_exp(exp_name):
    ''''
    Parse the names of genes we will need information on.
    '''
    print("parsing name...")
    parts = exp_name.split('+')
    perturbed_genes = []
    
    for part in parts:
        if part[1:].isdigit():
            perturbed_genes.append(part)
    print("returning: ", perturbed_genes)
    return perturbed_genes

def start():
    print("Data shape: ",  data.shape)
    print("Starting data processing...")
    genes = data.index.tolist()
    experiments = data.columns.tolist()


    # create empty feature matrix
    X = np.zeros((len(experiments), len(genes)))
    Y = np.zeros((len(experiments), len(genes)))
    print("X and Y matrices created with shapes:", X.shape, Y.shape)

    for exp_idx, exp_name in enumerate(experiments):
        print(f"Experiment {exp_idx}: {exp_name}")
        perturbed_genes = parse_exp(exp_name)
        
        # mark the genes perturbed in this experiment (features)
        for gene in perturbed_genes:
            if gene in genes:
                gene_idx = genes.index(gene)
                X[exp_idx, gene_idx] = 1
        Y[exp_idx, :] = data.iloc[:, exp_idx].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target matrix shape: {Y.shape}")
    
    X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    coefficients = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ Y
    
    print(f"Coefficient matrix shape: {coefficients.shape}")
    def predict_expression(genes_to_perturb):
        """
        Predict gene expression when specific genes are perturbed
        
        Parameters:
        genes_to_perturb: list of gene names to perturb (e.g., ['g0001', 'g0002'])
        
        Returns:
        Dictionary with predicted expression for all genes
        """
        # Create feature vector
        features = np.zeros(len(genes))
        for gene in genes_to_perturb:
            if gene in genes:
                gene_idx = genes.index(gene)
                features[gene_idx] = 1
        
        # Add bias term
        features_with_bias = np.append(features, 1)
        
        # Predict all gene expressions
        predictions = features_with_bias @ coefficients
        
        # Return as dictionary
        return {gene: pred for gene, pred in zip(genes, predictions)}
    
    genes_to_perturb = ['g0037', 'g0083']  # Replace with your genes of interest
    predictions = predict_expression(genes_to_perturb)

    print(f"\nPredicted expression when perturbing {genes_to_perturb}:")
    for gene, expr in list(predictions.items())[:5]:  # Show first 5 predictions
        print(f"{gene}: {expr:.6f}")
start()

