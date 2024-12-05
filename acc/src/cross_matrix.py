# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /home/u1/03_Programowanie/03Python/skrypty/acc/acc/src/cross_matrix.py
# Bytecode version: 3.11a7e (3495)
# Source timestamp: 2024-11-30 15:34:56 UTC (1732980896)

import sys
import copy
import numpy as np
import pandas as pd
predict_name = 'predicted'
true_name = 'true'


class RawData:
    """
    Stores classification results in a table with 2 columns:
    - First column: true values (actual classes)
    - Second column: predicted values (predicted classes)

    """

    def __init__(self, data, map_labels=None, default_label='cl'):
        """
        Input:
           - Columns must be in order [true_values, predicted, (label)]
           - Column names do not matter (data can be without column names)
           - Data can be a list, tuple, numpy array, or pandas DataFrame

        |  true  | predicted |    |  true  | predicted | label |
        |--------+-----------| or |--------+-----------+-------|
        |  int   |    int    |    |   int  |    int    |  rye  | 
        |  ...   |    ...    |    |   ...  |    ...    |  ...  | 

        Args:
            - data: Input data (list, tuple, numpy array, or DataFrame)
            - map_labels: Dictionary mapping class IDs to class names
            - default_label: Default prefix for class labels (e.g., "cl")
        """
        df = self._prepare_dataframe(data)
        df = self._clean_data(df)
        self.true_values = df.iloc[:, 0].astype(int).tolist()
        self.predicted = df.iloc[:, 1].astype(int).tolist()
        self.map_labels = self._generate_map_labels(df, map_labels, default_label)

    def _prepare_dataframe(self, data):
        """
        Converts input data to a pandas DataFrame and ensures it has 2 columns.

        Args:
            data: Input data (list, tuple, numpy array, or DataFrame)

        Returns:
            A pandas DataFrame with 2 columns.

        Raises:
            ValueError: If the input data does not have exactly 2 columns.
        """
        if isinstance(data, pd.DataFrame):
            df = data.reset_index(drop=True)
        elif isinstance(data, (list, tuple, np.ndarray)):
            df = pd.DataFrame(data).T
        else:
            raise ValueError('Data must be a DataFrame, list, tuple,                     or numpy array!')
        if df.shape[1] not in [2, 3]:
            raise ValueError('Data must have exactly 2 columns!')
        return df

    def _clean_data(self, df):
        """
        Cleans the data by converting columns to numbers and removing
        invalid rows.

        Args:
            df: A pandas DataFrame.

        Returns:
            A cleaned DataFrame with numeric columns and no invalid rows.
        """
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')

        df.dropna(inplace=True)

        # removes rows for containing predicted values that are not in true
        uniqe = set(df.iloc[:, 0])  # true
        idx = df.iloc[:, 1].isin(uniqe)
        df = df.loc[idx]
        return df

    def _generate_map_labels(self, df, map_labels, default_label):
        """
        Creates a map of class IDs to class labels.

        Args:
            - df: A cleaned DataFrame.
            - map_labels: A dictionary provided by the user (optional).
            - default_label: Default prefix for class labels.

        Returns:
            A dictionary mapping class IDs to class labels.
        """
        # all_classes = set(df.iloc[:, 0]).union(df.iloc[:, 1])
        all_classes = set(df.iloc[:, 0])

        if map_labels is not None:
            tmp = map_labels.copy()
            map_labels = {key: tmp[key] for key in all_classes}
            return map_labels

        # 3 columns: true, predict, short_label
        if df.shape[1] == 3:
            map_labels = dict(zip(df.iloc[:, 0], df.iloc[:, -1]))
            return map_labels


        if max(all_classes) > 9:
            n = 2
        else:
            n =1

        map_labels = {cls: f'{default_label}_{cls:0>{n}}'
                for cls in all_classes
                      }
        return map_labels

    def __repr__(self):
        result = []
        for cls, label in self.map_labels.items():
            result.append(f'  {cls} -> {label}')
        return '\n'.join(result)

class CrossMatrixRecognizer:
    """
    Utility class to recognize the type of a cross matrix.

    Methods:
        is_raw(df): Checks if the matrix is 'raw' (only numbers, no
                labels or sums).
        is_full(df): Checks if the matrix is 'full' (contains sums of rows
                and columns).
        is_cross(df): Checks if the matrix is 'cross' (contains labels,
                no sums)
    """

    @staticmethod
    def is_raw(df: pd.DataFrame) -> bool:
        """Check if the DataFrame is in raw cross format."""
        if isinstance(df, np.ndarray) and df.shape[0] == df.shape[1]:
            return True

        columns_are_numbers = pd.to_numeric(df.columns, errors='coerce').notna()
        index_are_numbers = pd.to_numeric(df.index, errors='coerce').notna()
        result = all(columns_are_numbers) and all(index_are_numbers)
        return result

    @staticmethod
    def is_cross(df: pd.DataFrame) -> bool:
        """Check if the DataFrame is in cross matrix format."""
        is_full = CrossMatrixRecognizer.is_full(df)
        is_raw = CrossMatrixRecognizer.is_raw(df)
        result = not is_full and (not is_raw)
        return result

    @staticmethod
    def is_full(df: pd.DataFrame) -> bool:
        """Check if the DataFrame is in full cross format."""
        cols_sum = df.iloc[:-1, :-1].sum(axis=0)
        rows_sum = df.iloc[:-1, :-1].sum(axis=1)
        correct_column_sums = all(df.iloc[-1, :-1] == cols_sum)
        correct_row_sums = all(df.iloc[:-1, -1] == rows_sum)
        result = correct_column_sums and correct_row_sums
        return result

class CrossMatrix:
    """
    Generates confusion matrices for classification results.

    Default layout of cross(confusion) matrix is:
      - rows: True classes (true labels).
      - columns: Predicted classes (predicted labels)

    Attributes:
        map_labels: Dictionary mapping class IDs to class names.
        true_values: List of true class IDs.
        predicted: List of predicted class IDs.
        cross_raw: Confusion matrix as raw numbers (no labels or summaries).
        cross: Confusion matrix with row and column descriptions.
        cross_full: Confusion matrix with descriptions and summaries.
    """

    def __init__(self, true_values, predicted, map_labels):
        """
        Initializes the CrossMatrix object.

        Args:
            map_labels: Dictionary {class_id: class_name, ...}.
            true_values: List of true class IDs.
            predicted: List of predicted class IDs.
        """
        self.true_values = true_values
        self.predicted = predicted
        self.map_labels = map_labels
        self.cross_raw = None
        self.cross = None
        self.cross_full = None
        self._generate_matrices()

    def _generate_matrices(self):
        """Generates all confusion matrix variants."""
        all_classes = sorted(set(self.true_values))
        tmp_cross = pd.crosstab(pd.Series(self.true_values, name='true'), pd.Series(self.predicted, name='predicted'), dropna=False)
        tmp_cross = tmp_cross.reindex(columns=all_classes, fill_value=0)
        self.cross_raw = tmp_cross.copy()
        labels = range(tmp_cross.shape[1])
        self.cross_raw.columns = labels
        self.cross_raw.index = labels
        self.cross_raw.columns.name = 'predicted'
        self.cross_raw.index.name = 'true'
        self.cross = self._add_labels(tmp_cross)
        self.cross_full = self._add_summaries(self.cross)

    def _add_labels(self, matrix):
        """Adds labels to rows and columns of the confusion matrix."""
        row_labels = [self.map_labels.get(i, f'Unknown_{i}0') for i in matrix.index]
        col_labels = [self.map_labels.get(i, f'Unknown_{i}0') for i in matrix.columns]
        return matrix.rename(index=dict(zip(matrix.index, row_labels)), columns=dict(zip(matrix.columns, col_labels)))

    def _add_summaries(self, matrix):
        """Adds summary rows and columns to the confusion matrix."""
        matrix_with_sums = matrix.copy()
        matrix_with_sums.loc['sums'] = matrix_with_sums.sum(axis=0)
        matrix_with_sums['sums'] = matrix_with_sums.sum(axis=1)
        return matrix_with_sums

    def __repr__(self):
        return self.cross_full.to_string(max_rows=5, max_cols=5) if self.cross_full is not None else 'Confusion matrix has not been generated yet.'

class CrossMatrixValidator:
    """
    Validates and processes a cross matrix.

    Attributes:
        data (pd.DataFrame): The original cross matrix (raw, cross, or full)
        type_cross (str): Detected type of the matrix ('raw', 'cross'
             or 'full')
        cross_raw (pd.DataFrame): Processed square matrix without labels
             or sums
        cross (pd.DataFrame): Matrix with labels
        cross_full (pd.DataFrame): Matrix with labels and sums.
    """
    pass
    pass
    pass
    pass

    def __init__(self, data, map_labels: dict=None, label_prefix='cl', scheme='normal', type_cross=None):
        """
        Initializes the validator with the input data and optional parameters.

        Args:
            data: The cross matrix, which can be a list of lists, np.array,
                   or pd.DataFrame.
            map_labels (dict, optional): Optional mapping for labels.
            label_prefix (str): Prefix for generating column and row labels
                   (e.g., 'cl' -> 'cl_01').
            scheme (str): Layout of rows and columns ('normal' or 'reverse').
            type_cross (str, optional): Predefined matrix type
                   ('raw', 'cross', or 'full').
        """
        self.scheme = scheme
        self.data = self._enter_data(data)
        self.map_labels = map_labels.copy() if map_labels else None
        self.label_prefix = label_prefix
        self.type_cross = type_cross if type_cross else self._detect_type()
        self.cross_raw = None
        self.cross_raw_sq = None
        self.cross = None
        self.cross_sq = None
        self.cross_full = None
        self.cross_full_sq = None
        self._process_matrices()

    def __repr__(self):
        return f'Type_cross: {self.type_cross}\nMap_labels: {self.map_labels}'

    def _process_matrices(self):
        """
        Processes the input data into different matrix forms.
        """
        self.map_labels = self._create_map_labels()
        if self.type_cross == 'raw':
            self.cross_raw = self.data.copy()
            self._cross_from_raw()
            self._full_from_cross()
        elif self.type_cross == 'cross':
            self._remap_labels()
            self._raw_from_cross()
            self._full_from_cross()
        elif self.type_cross == 'full':
            self._remap_labels()
            self._cross_from_full()
            self._raw_from_cross()

    def _enter_data(self, data) -> pd.DataFrame:
        """
        Converts the input data into a DataFrame and adjusts it based on
        the scheme.

        Args:
            data: Input data, which can be a list, np.array, or pd.DataFrame.

        Returns:
            pd.DataFrame: Adjusted DataFrame with integer values.
        """
        data = copy.deepcopy(data)
        if isinstance(data, (list, tuple, np.ndarray)):
            data = pd.DataFrame(data)
            data.columns = range(data.shape[1])
            data.index = range(data.shape[0])
            self.type_cross = 'raw'
        if self.scheme == 'reverse':
            data = data.T
        data.columns.name = 'predicted'
        data.index.name = 'true'
        return data.astype(int)

    def _detect_type(self) -> str:
        """
        Detects the type of the input matrix.

        Returns:
            str: The type ('raw', 'cross', 'full').
        """
        if CrossMatrixRecognizer.is_raw(self.data):
            return 'raw'
        if CrossMatrixRecognizer.is_full(self.data):
            return 'full'
        return 'cross'

    def _create_map_labels(self) -> dict:
        """
        Creates a mapping of numerical labels to string labels.

        Returns:
            dict: A dictionary mapping numerical labels to string labels.
        """
        if self.map_labels is not None:
            return self.map_labels
        if self.type_cross == 'raw':
            prefix = self.label_prefix
            n = self.data.shape[0]
            return {i: f'{prefix}_{i:0>2}' for i in range(1, n - 1)}

    def _remap_labels(self):
        """
        Remaps labels in the data using the mapping provided.
        """
        data = self.data.copy()
        if self.type_cross == 'full':
            data = data.iloc[:-1, :-1]
        try:
            if self.map_labels:
                data.columns = [self.map_labels[key] for key in data.columns]
                data.index = [self.map_labels[key] for key in data.index]
        except Exception:
            m1 = '\n\tCheck json data (map_labels)!'
            m2 = '\tClasses from this file do not match the data!!!\n'
            sys.exit(f'{m1}\n{m2}')
        data_sq = CrossMatrixValidator._make_matrix_square(data, self.scheme)
        if self.type_cross == 'cross':
            self.cross = data
            self.cross_sq = data_sq
        elif self.type_cross == 'full':
            self.cross_full = self._add_sums_cols_rows(data)
            self.cross_full_sq = self._add_sums_cols_rows(data_sq)

    def _raw_from_cross(self):
        """
        Creates a square version of the raw matrix without labels or sums.
        """
        cross_raw = self.cross_sq.copy()
        cross_raw.index = range(len(cross_raw.index))
        cross_raw.columns = range(len(cross_raw.columns))
        cross_raw.columns.name = 'predicted'
        cross_raw.index.name = 'true'
        self.cross_raw = cross_raw

    def _full_from_cross(self):
        """
        Creates the full cross matrix with sums added to rows and columns.
        """
        self.cross_full = self._add_sums_cols_rows(self.cross)
        self.cross_full_sq = self._add_sums_cols_rows(self.cross_sq)

    def _cross_from_raw(self):
        """
        Creates a labeled cross matrix from a raw matrix.
        """
        cross = self.cross_raw.copy()
        map_labels = self.map_labels
        names = [map_labels[i] for i in range(1, cross.shape[0] - 1)]
        cross.columns = names
        cross.index = names
        cross.columns.name = 'predicted'
        cross.index.name = 'true'
        self.cross = cross
        self.cross_sq = cross

    def _cross_from_full(self):
        """
        Extracts the cross matrix from a full matrix by removing sums.
        """
        self.cross = self.cross_full.iloc[:-1, :-1]
        self.cross_sq = self.cross_full_sq.iloc[:-1, :-1]

    @staticmethod
    def _make_matrix_square(matrix: pd.DataFrame, scheme: str) -> pd.DataFrame:
        """
        Ensures the matrix is square by reindexing rows and columns.

        Args:
            matrix (pd.DataFrame): The input matrix.
            scheme (str): Layout scheme ('normal' or 'reverse').

        Returns:
            pd.DataFrame: A square matrix.
        """
        matrix = matrix.copy()
        if scheme == 'normal':
            matrix = matrix.reindex(columns=matrix.index, fill_value=0)
        else:
            matrix = matrix.reindex(index=matrix.columns, fill_value=0)
        return matrix

    @staticmethod
    def _add_sums_cols_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds sums to the rows and columns of the matrix.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with sums added.
        """
        df = df.copy()
        df.loc[:, 'sums'] = df.sum(axis=1)
        df.loc['sums', :] = df.sum(axis=0)
        df.columns.name = 'predicted'
        df.index.name = 'true'
        return df.astype(int)
