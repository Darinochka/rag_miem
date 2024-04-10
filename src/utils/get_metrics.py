import pandas as pd
import numpy as np
import os
import argparse
from typing import List, Optional


def calculate_ndcg(
    true_relevance: List[int], relevances: List[int], k: Optional[int] = None
) -> float:
    """
    Calculate NDCG for a single list of relevances.
    """
    if k is not None:
        relevances = relevances[:k]
    relevant_array = np.array(relevances)
    dcg = np.sum(relevant_array / np.log2(np.arange(2, relevant_array.size + 2)))
    idcg = np.sum(true_relevance / np.log2(np.arange(2, relevant_array.size + 2)))
    return dcg / idcg if idcg > 0 else 0


def calculate_reciprocal_rank(relevances: List[int]) -> float:
    """
    Calculate Reciprocal Rank (RR) for a single list of relevances.
    """
    relevant_array = np.array(relevances)
    try:
        # Get the rank (index + 1) of the first relevant item (where relevance > 0)
        rank = np.where(relevant_array > 0)[0][0] + 1
        return 1.0 / rank
    except IndexError:
        # If there is no relevant item, return 0
        return 0


def calculate_average_precision(
    relevances: List[int], k: Optional[int] = None
) -> float:
    """
    Calculate Average Precision (AP) for a single list of relevances.
    """
    if k is not None:
        relevances = relevances[:k]
    relevances = np.array(relevances)
    cumsum = np.cumsum(relevances)
    precision_at_k = cumsum / (np.arange(len(relevances)) + 1)
    return (
        np.sum(precision_at_k * relevances) / np.sum(relevances)
        if np.sum(relevances) > 0
        else 0
    )


def process_folder(folder_path: str) -> None:
    """
    Process each CSV file in the specified folder.
    """
    print("model,Average NDCG,mAP,mRR")
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Replace 9 with 1 in rank columns
            rank_columns = [col for col in df.columns if "rank" in col]
            if not rank_columns:
                continue

            df[rank_columns] = df[rank_columns].replace(9, 1)

            relevances = df[rank_columns].values
            true_relevance = np.ones(relevances.shape[1])
            ndcg_scores = [calculate_ndcg(true_relevance, row) for row in relevances]
            ap_scores = [calculate_average_precision(row) for row in relevances]
            rr_scores = [calculate_reciprocal_rank(row) for row in relevances]

            avg_ndcg = np.mean(ndcg_scores)
            map = np.mean(ap_scores)
            mrr = np.mean(rr_scores)

            print(f"{filename[:-4]},{avg_ndcg:.4f},{map:.4f},{mrr:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate NDCG and Precision metrics for CSV files in a folder."
    )
    parser.add_argument(
        "--folder-path", type=str, help="Path to the folder containing CSV files"
    )

    args = parser.parse_args()
    process_folder(args.folder_path)


if __name__ == "__main__":
    main()
