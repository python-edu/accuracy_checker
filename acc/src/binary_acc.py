# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


"""
Module version: {0}.

This class creates a binary indicators table (TP, TN, FP, FN), called
'bin_tab', based on a confusion matrix.

Note:
    The confusion matrix must not include row and column summaries!

Attributes:
    layout (str): Defines the layout of the binary table.
                  - "h" (default): Horizontal layout.
                  - "v": Vertical layout.
    row_names (tuple): Names of the rows in the binary table. Default is
                  ("TP", "TN", "FP", "FN").

Usage:
    # Create an instance of BinTable
    bt = BinTable(layout="h")

    # Example confusion matrix
    cross_matrix = pd.DataFrame({
        'water': [21, 5, 7],
        'forest': [6, 31, 2],
        'urban': [0, 1, 22]
    }, index=['water', 'forest', 'urban'])

    # Generate the binary table
    binary_table = bt(cross_matrix)

    # Result:
          water  forest  urban
    TP      21      31     22
    TN      56      50     63
    FP       6       6      9
    FN      12       8      1

    # For vertical layout:
    bt = BinTable(layout="v")
    binary_table = bt(cross_matrix)

    # Result:
             TP  TN  FP  FN
    water    21  56   6  12
    forest   31  50   6   8
    urban    22  63   9   1
"""


class BinTable:
    """
    Class for creating a binary indicators table (TP, TN, FP, FN) from a
    confusion matrix.

    Attributes:
        layout (str): Layout of the resulting binary table.
                      "h" (default) for horizontal, "v" for vertical.
        row_names (tuple): Names of the rows in the binary table.
                           Default is ("TP", "TN", "FP", "FN").
        data (pd.DataFrame or None): The confusion matrix, if provided during
                           initialization.
    """

    def __init__(self,
                 data_frame=None,
                 layout="h",
                 row_names=("TP", "TN", "FP", "FN")
                 ):
        """
        Initialize the BinTable instance.

        Args:
            data_frame (pd.DataFrame or None): Confusion matrix as a DataFrame.
            layout (str): Layout of the resulting table. "h" for horizontal,
                          "v" for vertical.
            row_names (tuple): Names of the rows in the binary table.
        """
        self.layout = layout
        self.row_names = row_names
        self.data = None

        if data_frame is not None and self._valid_matrix(data_frame):
            self.data = data_frame
            self.__call__(data_frame, row_names)

    def _valid_matrix(self, df):
        """
        Validate that the input is a square confusion matrix.

        Args:
            df (pd.DataFrame): The confusion matrix to validate.

        Raises:
            ValueError: If the input is not a square DataFrame.
        """
        if df is None or not isinstance(df, (pd.DataFrame, np.ndarray)):
            raise ValueError("No data provided. "
                             "A square confusion matrix is required!"
                             )
        if df.shape[0] != df.shape[1]:
            raise ValueError(
                "The confusion matrix must be square! "
                f"Current shape: {df.shape[0]}x{df.shape[1]}"
            )

    def __call__(self, data_frame=None, row_names=None):
        """
        Generate the binary table from the confusion matrix.

        Args:
            data_frame (pd.DataFrame): The confusion matrix.
            row_names (list or tuple, optional): Row names for the binary table

        Returns:
            pd.DataFrame: Binary indicators table.
        """
        self._valid_matrix(data_frame)
        if row_names is None:
            row_names = self.row_names

        array, col_names = self._get_data(data_frame)
        bin_tab = self._bin_table(array, col_names, row_names)
        if self.layout == "v":
            bin_tab = bin_tab.T
        return bin_tab

    @staticmethod
    def _get_data(data_frame):
        """
        Convert the DataFrame to a NumPy array and extract column names.

        Args:
            data_frame (pd.DataFrame): The input DataFrame.

        Returns:
            tuple: NumPy array and list of column names.
        """
        return data_frame.to_numpy(), tuple(data_frame.columns.to_list())

    def _bin_table(self, array, col_names, row_names):
        """
        Create the binary table from the confusion matrix.

        Args:
            array (np.ndarray): Confusion matrix as a NumPy array.
            col_names (list): Names of the columns.
            row_names (list): Names of the rows.

        Returns:
            pd.DataFrame: Binary indicators table.
        """
        results = {}

        for i in range(array.shape[0]):
            tp = array[i, i]

            tmp = np.delete(array, i, axis=1)
            tmp = np.delete(tmp, i, axis=0)  # Remove row of the current class
            tn = tmp.sum()

            row = np.delete(array[i, :], i)  # Exclude current column from row
            fn = row.sum()

            col = np.delete(array[:, i], i)  # Exclude current row from column
            fp = col.sum()

            results[col_names[i]] = [tp, tn, fp, fn]

        return pd.DataFrame(results, index=row_names).astype("int")

    def __repr__(self):
        """
        String representation of the binary table if the data is available.

        Returns:
            str: String representation of the data.
        """
        if hasattr(self, "data") and self.data is not None:
            return self.data.to_string()
        return "BinTable instance with no data."
