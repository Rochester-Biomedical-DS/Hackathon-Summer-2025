import csv

test_set_path = 'data/test_set.csv'
prediction_path = 'prediction/prediction.csv'
output_path = 'prediction/prediction_sorted.csv'

# Read the order from test_set.csv (single column: perturbation)
with open(test_set_path, newline='') as f:
    reader = csv.reader(f)
    test_order = [row[0] for row in reader if row]

# Read predictions into a list of rows and a dict by perturbation
with open(prediction_path, newline='') as f:
    reader = csv.reader(f)
    pred_header = next(reader)
    predictions = {}
    for row in reader:
        if row:
            perturbation = row[1]
            if perturbation not in predictions:
                predictions[perturbation] = []
            predictions[perturbation].append(row)

# Write predictions in the order of test_set.csv, including header
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(pred_header)
    for perturbation in test_order:
        if perturbation in predictions:
            for row in predictions[perturbation]:
                writer.writerow(row)