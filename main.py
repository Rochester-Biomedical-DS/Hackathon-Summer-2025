import csv
import os
from pseudoinverse import *

input_csv = 'data/test_set.csv'
output_csv = 'prediction/prediction.csv'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Train model once
coefficients, gene_names = train_with_pseudoinverse('data/train_set.csv')

with open(input_csv, newline='') as infile, open(output_csv, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    # Write header
    writer.writerow(['gene', 'perturbation', 'expression'])
    for row in reader:
        print("Processing row:", row)
        if not row or len(row[0].split('+')) != 2:
            continue  # skip incomplete rows
        perturbation = row[0]
        gene1, gene2 = perturbation.split('+')
        predictions = predict_with_pseudoinverse(coefficients, gene_names, [gene1, gene2])
        for gene, value in predictions.items():
            writer.writerow([gene, perturbation, value])
    print("PROCESS FINISHED")