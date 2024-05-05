"""Utility functions for saving predictions for submission to Kaggle."""

import csv
import os

import numpy as np


def write_csv(file_path: str, y_list: np.ndarray):
    """Write a CSV file.

    Parameters:
        file_path: name of the file to save
        y_list: y predictions
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    solution_rows = [("id", "category")] + [(i, y) for (i, y) in enumerate(y_list)]
    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(solution_rows)


def output_submission_csv(output_file_path: str, y_test: np.ndarray):
    """Save predictions for Kaggle submission.

    Parameters:
        output_file_path: name of the file to save
        y_test: y predictions
    """
    write_csv(output_file_path, y_test)
