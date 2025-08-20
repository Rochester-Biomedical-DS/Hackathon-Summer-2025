import csv
import os
import sys
from pseudoinverse import run_pseudo

# Import run_pseudo from pseudoinverse.py

input_csv = 'data/test_set.csv'
output_csv = 'prediction/prediction.csv'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

with open(input_csv, newline='') as infile, open(output_csv, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    # Write header
    writer.writerow(['gene', 'gene1', 'gene2', 'prediction'])
    for row in reader:
        if len(row) < 2:
            continue  # skip incomplete rows
        gene1, gene2 = row.split('+')[0], row.split('+')[1]
        prediction = run_pseudo(gene1, gene2)
        writer.writerow([f"{gene1}_{gene2}", gene1, gene2, prediction])