import pandas as pd
import sys
import traceback

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
    
    if phase_codename == "final":
        print("Evaluating for Final Phase")
        try:
            output["result"] = [
                {
                    "test_split": {
                        'Accuracy Score': accuracy,
                        'Precision' : precision,
                        'Recall' : recall,
                        'F1 score' : f1_score,
                    }
                }
            ]
            
            return output
        except Exception as e:
            sys.stderr.write(traceback.format_exc())
            return e
