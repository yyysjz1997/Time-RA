import json
import numpy as np
from tqdm import tqdm
from metrics.similarity import cos_similarity, tfidf_similarity, levenshtein_similarity, token_sequence_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import argparse

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def compare_entries(json1, json2, model_name):

    common_keys = set(json1.keys()) & set(json2.keys())
    print(f"Number of common keys: {len(common_keys)}")

    cosine_total = tfidf_total = levenshtein_total = token_seq_total = 0
    actionid_match_total = label_match_total = 0

    pred_labels, true_labels = [], []
    pred_action_ids, true_action_ids = [], []

    all_action_ids = list(set([json1[k]["ActionID"] for k in common_keys] + [json2[k]["ActionID"] for k in common_keys]))
    le = LabelEncoder()
    le.fit(all_action_ids)

    pbar = tqdm(sorted(common_keys, key=int), desc="Comparing entries")

    for idx, key in enumerate(pbar, 1):
        item1 = json1[key]
        item2 = json2[key]

        actionid_match_total += int(item1["ActionID"] == item2["ActionID"])
        label_match_total += int((item1["ActionID"] == 0 and item2["Label"] == 'Normal') or 
                                 (item1["ActionID"] != 0 and item2["Label"] == 'Anomaly'))

        pred_label = 0 if item1["ActionID"] == 0 else 1
        true_label = 0 if item2["Label"] == 'Normal' else 1

        pred_labels.append(pred_label)
        true_labels.append(true_label)

        pred_action_ids.append(item1["ActionID"])
        true_action_ids.append(item2["ActionID"])

        cosine_total += float(cos_similarity([item2["Thought"], item1["Thought"]]))
        tfidf_total += tfidf_similarity([item2["Thought"], item1["Thought"]])
        levenshtein_total += levenshtein_similarity([item2["Thought"], item1["Thought"]])
        token_seq_total += token_sequence_similarity([item2["Thought"], item1["Thought"]])

        label_f1 = f1_score(true_labels[:idx], pred_labels[:idx], zero_division=0)
        act_f1 = f1_score(
            le.transform(true_action_ids[:idx]), 
            le.transform(pred_action_ids[:idx]),
            average='macro', zero_division=0
        )

        pbar.set_postfix({
            "Cosine": f"{cosine_total / idx:.3f}",
            "TFIDF": f"{tfidf_total / idx:.3f}",
            "Lev": f"{levenshtein_total / idx:.3f}",
            "TokSeq": f"{token_seq_total / idx:.3f}",
            "F1(Label)": f"{label_f1:.3f}",
            "F1(AID)": f"{act_f1:.3f}"
        })

    label_precision = precision_score(true_labels, pred_labels, zero_division=0)
    label_recall = recall_score(true_labels, pred_labels, zero_division=0)
    label_f1 = f1_score(true_labels, pred_labels, zero_division=0)

    true_ids_encoded = le.transform(true_action_ids)
    pred_ids_encoded = le.transform(pred_action_ids)

    action_precision = precision_score(true_ids_encoded, pred_ids_encoded, average='macro', zero_division=0)
    action_recall = recall_score(true_ids_encoded, pred_ids_encoded, average='macro', zero_division=0)
    action_f1 = f1_score(true_ids_encoded, pred_ids_encoded, average='macro', zero_division=0)

    print("\nFinal Average Results:")
    print(f"Cosine Similarity: {cosine_total / len(common_keys):.4f}")
    print(f"TFIDF Similarity: {tfidf_total / len(common_keys):.4f}")
    print(f"Levenshtein Similarity: {levenshtein_total / len(common_keys):.4f}")
    print(f"Token Sequence Similarity: {token_seq_total / len(common_keys):.4f}")
    print(f"ActionID Match: {actionid_match_total / len(common_keys):.2%}")
    print(f"Label Match: {label_match_total / len(common_keys):.2%}")

    print("\nLabel Metrics:")
    print(f"Precision: {label_precision:.4f}")
    print(f"Recall:    {label_recall:.4f}")
    print(f"F1-score:  {label_f1:.4f}")

    print("\nActionID Metrics:")
    print(f"Precision: {action_precision:.4f}")
    print(f"Recall:    {action_recall:.4f}")
    print(f"F1-score:  {action_f1:.4f}")

    np.save(f"result/{model_name}_pred_labels.npy", np.array(pred_labels))
    np.save(f"result/{model_name}_true_labels.npy", np.array(true_labels))
    np.save(f"result/{model_name}_pred_action_ids.npy", np.array(pred_action_ids))
    np.save(f"result/{model_name}_true_action_ids.npy", np.array(true_action_ids))
    
    with open(f"result/{model_name}_results.txt", 'w') as f:
        print("\nFinal Average Results:", file=f)
        print(f"Cosine Similarity: {cosine_total / len(common_keys):.4f}", file=f)
        print(f"TFIDF Similarity: {tfidf_total / len(common_keys):.4f}", file=f)
        print(f"Levenshtein Similarity: {levenshtein_total / len(common_keys):.4f}", file=f)
        print(f"Token Sequence Similarity: {token_seq_total / len(common_keys):.4f}", file=f)
        print(f"ActionID Match: {actionid_match_total / len(common_keys):.2%}", file=f)
        print(f"Label Match: {label_match_total / len(common_keys):.2%}", file=f)

        print("\nLabel Metrics:", file=f)
        print(f"Precision: {label_precision:.4f}", file=f)
        print(f"Recall:    {label_recall:.4f}", file=f)
        print(f"F1-score:  {label_f1:.4f}", file=f)

        print("\nActionID Metrics:", file=f)
        print(f"Precision: {action_precision:.4f}", file=f)
        print(f"Recall:    {action_recall:.4f}", file=f)
        print(f"F1-score:  {action_f1:.4f}", file=f)


def parse_metrics_to_string(filepath):
    """
    Reads a text file containing performance metrics, extracts specific values,
    and returns them as a formatted string.

    Args:
        filepath (str): The path to the input text file.

    Returns:
        str: A formatted string containing the extracted metrics,
             or an error message if the file cannot be read or parsed.
    """
    metrics = {}
    last_header = ""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if "Label Match:" in line:
                metrics['Label Match'] = float(line.split(':')[1].strip().replace('%', '')) / 100
            elif "Precision:" in line and "Label Metrics" in last_header:
                metrics['Label Metrics Precision'] = float(line.split(':')[1].strip())
            elif "Recall:" in line and "Label Metrics" in last_header:
                metrics['Label Metrics Recall'] = float(line.split(':')[1].strip())
            elif "F1-score:" in line and "Label Metrics" in last_header:
                metrics['Label Metrics F1-score'] = float(line.split(':')[1].strip())
            elif "ActionID Match:" in line:
                metrics['ActionID Match'] = float(line.split(':')[1].strip().replace('%', '')) / 100
            elif "Precision:" in line and "ActionID Metrics" in last_header:
                metrics['ActionID Metrics Precision'] = float(line.split(':')[1].strip())
            elif "Recall:" in line and "ActionID Metrics" in last_header:
                metrics['ActionID Metrics Recall'] = float(line.split(':')[1].strip())
            elif "F1-score:" in line and "ActionID Metrics" in last_header:
                metrics['ActionID Metrics F1-score'] = float(line.split(':')[1].strip())
            elif "Cosine Similarity:" in line:
                metrics['Cosine'] = float(line.split(':')[1].strip())
            elif "TFIDF Similarity:" in line:
                metrics['TFIDF'] = float(line.split(':')[1].strip())
            elif "Levenshtein Similarity:" in line:
                metrics['Levenshtein'] = float(line.split(':')[1].strip())
            elif "Token Sequence Similarity:" in line:
                metrics['Token'] = float(line.split(':')[1].strip())

            # Keep track of the last header encountered to distinguish Precision/Recall/F1
            if "Label Metrics" in line:
                last_header = "Label Metrics"
            elif "ActionID Metrics" in line:
                last_header = "ActionID Metrics"

    except FileNotFoundError:
        return f"Error: File not found at {filepath}"
    except Exception as e:
        return f"Error parsing file: {e}"

    # Format the output string using the extracted metrics
    try:
        res = (
            f"{metrics['Label Match']:.4f}&"
            f"{metrics['Label Metrics Precision']:.4f}&"
            f"{metrics['Label Metrics Recall']:.4f}&"
            f"{metrics['Label Metrics F1-score']:.4f}&"
            f"{metrics['ActionID Match']:.4f}&"
            f"{metrics['ActionID Metrics Precision']:.4f}&"
            f"{metrics['ActionID Metrics Recall']:.4f}&"
            f"{metrics['ActionID Metrics F1-score']:.4f}&"
            f"{metrics['Cosine']:.4f}&"
            f"{metrics['TFIDF']:.4f}&"
            f"{metrics['Levenshtein']:.4f}&"
            f"{metrics['Token']:.4f}"
        )
        return res
    except KeyError as e:
        return f"Error: Missing expected metric in file: {e}. Please ensure all required lines are present."
import os

def read_all_txt_files(directory_path):
    """
    Reads and prints the content of all .txt files in the specified directory.

    Args:
        directory_path (str): The path to the directory containing the .txt files.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return []

    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    return txt_files


if __name__ == "__main__":
    directory = "result"
    file_names = read_all_txt_files(directory)
    for filename in file_names:
        file_path = os.path.join(directory, filename)
        str_res = parse_metrics_to_string(file_path) + "\\\\"
        print(filename, str_res)
