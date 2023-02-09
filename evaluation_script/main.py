import pandas as pd

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    # Load the ground truth labels and predictions into pandas dataframes
    ground_truth = pd.read_csv(test_annotation_file)
    prediction = pd.read_csv(user_submission_file)

    # Merge the two dataframes on the "id" and "word" columns
    merged_df = pd.merge(ground_truth, prediction, on=["id", "POI/street"], how="outer", indicator=True)
    merged_df22 = pd.merge(ground_truth, prediction, on=["id", "POI/street"])

    # Calculate the number of true positive, false positive, and false negative examples
    true_positive = merged_df[merged_df._merge == "both"].shape[0]
    false_positive = merged_df[merged_df._merge == "right_only"].shape[0]
    false_negative = merged_df[merged_df._merge == "left_only"].shape[0]
    correct_predictions = merged_df22.shape[0]

    # Calculate precision, recall, and F1-score
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)

    total_predictions = ground_truth.shape[0]
    accuracy = correct_predictions / total_predictions

    output = {}
    if phase_codename == "train":
        print("Evaluating for Train Phase")
        output["result"] = [
            {
                "train_split": {
                    'Accuracy Score': accuracy,
                    'Precision' : precision,
                    'Recall' : recall,
                    'F1 score' : f1_score,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["train_split"]
        print("Completed evaluation for Train Phase")
    return output










# import csv

# def accuracy(ground_truth, predictions):
#     correct = 0
#     for gt, pred in zip(ground_truth, predictions):
#         if gt[1] == pred[1]:
#             correct += 1
#     return correct / len(ground_truth)

# def evaluate(dataset, test_annotation_file, user_submission_file, phase_codename, **kwargs):
#     accuracy_sum = 0
#     for item in dataset:
#         ground_truth = item[1]
#         item_predictions = user_submission_file[item[0]]
#         accuracy_sum += accuracy(ground_truth, item_predictions)

#     ground_truth = {}
#     with open(test_annotation_file) as f:
#         reader = csv.reader(f)
#         for row in reader:
#             if row[0] not in ground_truth:
#                 ground_truth[row[0]] = []
#             ground_truth[row[0]].append(row)
    
#     predictions = {}
#     with open(user_submission_file) as f:
#         reader = csv.reader(f)
#         for row in reader:
#             if row[0] not in predictions:
#                 predictions[row[0]] = []
#             predictions[row[0]].append(row)
    
#     dataset = list(ground_truth.items())
#     akurasi = accuracy_sum / len(dataset)

#     output = {}
#     if phase_codename == "train":
#         print("Evaluating for Dev Phase")
#         output["result"] = [
#             {
#                 "train_split": {
#                     'Score': akurasi,
#                 }
#             }
#         ]
#         # To display the results in the result file
#         output["submission_result"] = output["result"][0]["train_split"]
#         print("Completed evaluation for Dev Phase")
#     return output

        
#     # dataset = list(ground_truth.items())


#     # accuracy = evaluate(dataset, predictions)
#     # print('Accuracy:', accuracy)


