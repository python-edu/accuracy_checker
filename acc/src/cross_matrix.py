import copy
import numpy as np
import pandas as pd

predict_name = "predicted"
true_name = "true"


class RawData:
    """
    Stores classification results in a table with two or three columns:
    - Column 1: True values (actual classes).
    - Column 2: Predicted values (predicted classes).
    - (Optional) Column 3: Short labels or custom labels.

    This class preprocesses the input data, validates it, and generates
    mappings for class IDs to readable labels.

    Attributes:
        true_values (list[int]): List of true class IDs (numeric).
        predicted (list[int]): List of predicted class IDs (numeric).
        map_labels (dict): Mapping of class IDs to readable labels.
    """

    def __init__(self, data, map_labels=None, default_label="cl"):
        """
        Initializes the RawData object with classification results.

        Input:
           - Columns must be in order [true_values, predicted_values, (label)].
           - Column names do not matter (data can be without column names).
           - Data can be a list, tuple, numpy array, or pandas DataFrame.

        Example input table:
        |  true  | predicted |    |  true  | predicted |  label  |
        |--------+-----------| or |--------+-----------+---------|
        |   int  |    int    |    |   int  |    int    |   str   |
        |  ...   |    ...    |    |   ...  |    ...    |   ...   |

        Args:
            data: Input data in one of the following formats:
                  - pandas DataFrame.
                  - list or tuple of lists/tuples.
                  - numpy array.
            map_labels: (Optional) A dictionary mapping class IDs to custom
                        labels. If not provided, labels are generated
                        automatically.
            default_label: (Optional) Prefix used for generating default
                           labels. Defaults to "cl".

        Raises:
            ValueError: If the input data is not in the expected format.
        """
        df = self._prepare_dataframe(data)
        df = self._clean_data(df)
        self.true_values = df.iloc[:, 0].astype(int).tolist()
        self.predicted = df.iloc[:, 1].astype(int).tolist()
        self.map_labels = self._generate_map_labels(df,
                                                    map_labels,
                                                    default_label
                                                    )

    def _prepare_dataframe(self, data):
        """
        Converts input data into a pandas DataFrame and ensures it has
        two or three columns.

        Args:
            data: Input data (DataFrame, list, tuple, or numpy array).

        Returns:
            pd.DataFrame: A DataFrame containing the processed data.

        Raises:
            ValueError: If the input data does not have the correct number
                        of columns.
        """
        if isinstance(data, pd.DataFrame):
            df = data.reset_index(drop=True)
        elif isinstance(data, (list, tuple, np.ndarray)):
            df = pd.DataFrame(data).T
        else:
            raise ValueError(
                "Input data must be a DataFrame, list, tuple, or numpy array!"
            )
        if df.shape[1] not in [2, 3]:
            raise ValueError("Input data must have exactly 2 or 3 columns!")
        return df

    def _clean_data(self, df):
        """
        Cleans the input DataFrame by converting values to numeric types
        and removing invalid rows.

        Args:
            df: A pandas DataFrame.

        Returns:
            pd.DataFrame: A cleaned DataFrame with valid numeric rows only.
        """
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        df.dropna(inplace=True)

        # Remove rows where predicted values are not in the set of true values
        true_classes = set(df.iloc[:, 0])  # Unique true values
        df = df[df.iloc[:, 1].isin(true_classes)]
        return df

    def _generate_map_labels(self, df, map_labels, default_label):
        """
        Generates a mapping of class IDs to readable labels.

        Args:
            df: A cleaned DataFrame.
            map_labels: (Optional) Dictionary mapping class IDs to labels.
            default_label: Default prefix for generating class labels.

        Returns:
            dict: A dictionary mapping class IDs to labels.
        """
        all_classes = set(df.iloc[:, 0])

        # Use the provided map_labels if available
        if map_labels is not None:
            return {cls: map_labels[cls] for cls in all_classes}

        # If a third column exists, use it as class labels
        if df.shape[1] == 3:
            return dict(zip(df.iloc[:, 0], df.iloc[:, -1]))

        # Generate default labels
        label_width = 2 if max(all_classes) > 9 else 1
        return {
            cls: f"{default_label}_{cls:0>{label_width}}"
            for cls in all_classes
        }

    def __repr__(self):
        """
        Returns a string representation of the class-to-label mapping.

        Returns:
            str: A formatted string of class IDs and their labels.
        """
        return "\n".join(f"  {cls} -> {label}"
                         for cls, label in self.map_labels.items()
                         )


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

        columns_are_numbers = pd.to_numeric(df.columns,
                                            errors="coerce").notna()
        index_are_numbers = pd.to_numeric(df.index, errors="coerce").notna()
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

    Default layout of the cross (confusion) matrix:
      - Rows: True classes (true labels).
      - Columns: Predicted classes (predicted labels).

    Attributes:
        true_values (list[int]): List of true class IDs.
        predicted (list[int]): List of predicted class IDs.
        map_labels (dict): Dictionary mapping class IDs to class names.
        cross_raw (pd.DataFrame): Confusion matrix with numeric values only.
        cross (pd.DataFrame): Confusion matrix with row and column descriptions
        cross_full (pd.DataFrame): Confusion matrix with row/column
                                   descriptions and summary rows/columns.
    """

    def __init__(self, true_values, predicted, map_labels):
        """
        Initializes the CrossMatrix object.

        Args:
            true_values (list[int]): List of true class IDs.
            predicted (list[int]): List of predicted class IDs.
            map_labels (dict): Dictionary mapping class IDs to class names.
        """
        self.true_values = true_values
        self.predicted = predicted
        self.map_labels = map_labels
        self.cross_raw = None
        self.cross = None
        self.cross_full = None
        self._generate_matrices()

    def _generate_matrices(self):
        """
        Generates the three variants of the confusion matrix:
        - cross_raw: Matrix with numeric values only.
        - cross: Matrix with row and column descriptions.
        - cross_full: Matrix with descriptions and summary rows/columns.
        """
        # Identify all classes from true values
        all_classes = sorted(set(self.true_values))

        # Generate raw confusion matrix (numeric only)
        tmp_cross = pd.crosstab(
            pd.Series(self.true_values, name="true"),
            pd.Series(self.predicted, name="predicted"),
            dropna=False,
        )
        tmp_cross = tmp_cross.reindex(columns=all_classes, fill_value=0)

        # Save raw matrix
        self.cross_raw = tmp_cross.copy()
        labels = range(tmp_cross.shape[0])
        self.cross_raw.columns = labels
        self.cross_raw.index = labels
        self.cross_raw.columns.name = "predicted"
        self.cross_raw.index.name = "true"

        # Generate labeled and full matrices
        self.cross = self._add_labels(tmp_cross)
        self.cross_full = self._add_summaries(self.cross)

    def _add_labels(self, matrix):
        """
        Adds labels to the rows and columns of the confusion matrix.

        Args:
            matrix (pd.DataFrame): A confusion matrix with numeric values only.

        Returns:
            pd.DataFrame: A matrix with row and column descriptions.
        """
        max_index_value = max(matrix.index)
        label_width = 2 if max_index_value > 9 else 1

        # Generate row and column labels
        row_labels = [
            self.map_labels.get(i, f"Unknown_{i:0>{label_width}}")
            for i in matrix.index
        ]
        col_labels = [
            self.map_labels.get(i, f"Unknown_{i:0>{label_width}}")
            for i in matrix.columns
        ]

        # Rename rows and columns
        return matrix.rename(
            index=dict(zip(matrix.index, row_labels)),
            columns=dict(zip(matrix.columns, col_labels)),
        )

    def _add_summaries(self, matrix):
        """
        Adds summary rows and columns to the confusion matrix.

        Args:
            matrix (pd.DataFrame): A confusion matrix with row/column
                                   descriptions.

        Returns:
            pd.DataFrame: A matrix with additional summary rows and columns.
        """
        matrix_with_sums = matrix.copy()

        # Add summary row and column
        matrix_with_sums.loc["sums"] = matrix_with_sums.sum(axis=0)
        matrix_with_sums["sums"] = matrix_with_sums.sum(axis=1)

        return matrix_with_sums

    def __repr__(self):
        """
        Returns a string representation of the full confusion matrix.

        Returns:
            str: A formatted string of the confusion matrix or a placeholder
                 message if the matrix has not been generated.
        """
        return (
            self.cross_full.to_string(max_rows=5, max_cols=5)
            if self.cross_full is not None
            else "Confusion matrix has not been generated yet."
        )


class CrossMatrixValidator:
    """
    Validates and processes a cross matrix.

    This class identifies the type of the matrix (raw, cross, or full) and
    converts it into different standardized forms, including labeled and
    full matrices with sums.

    Attributes:
        data (pd.DataFrame): The original input matrix.
        type_cross (str): Detected type of the matrix ('raw', 'cross',
                          or 'full').
        map_labels (dict): Mapping of numerical labels to string labels.
        label_prefix (str): Prefix for auto-generated labels.
        cross_raw (pd.DataFrame): Square numeric matrix without labels or sums.
        cross (pd.DataFrame): Matrix with row/column labels.
        cross_full (pd.DataFrame): Matrix with labels and summary rows/columns.
        cross_as (pd.DataFrame): Asymmetric cross matrix if applicable.
        cross_full_as (pd.DataFrame): Full asymmetric cross matrix with sums.
    """

    def __init__(
        self,
        data,
        map_labels: dict = None,
        label_prefix="cl",
        scheme="normal",
        type_cross=None,
    ):
        """
        Initializes the CrossMatrixValidator object with input data.

        Args:
            data: The input cross matrix. Can be a list of lists, numpy array,
                  or pandas DataFrame.
            map_labels (dict, optional): Optional mapping of numerical labels
                                         to string labels.
            label_prefix (str): Prefix for auto-generated labels
                                (e.g., 'cl_01').
            scheme (str): Layout of rows and columns ('normal' or 'reverse').
            type_cross (str, optional): Predefined type of the matrix
                                        ('raw', 'cross', or 'full').
        """
        self.scheme = scheme
        self.data = self._enter_data(data)
        self.map_labels = map_labels.copy() if map_labels else None
        self.map_labels_reversed = (dict(zip(map_labels.values(),
                                            map_labels.keys())) if map_labels
                                    else None)
        self.label_prefix = label_prefix
        self.type_cross = type_cross if type_cross else self._detect_type()
        self.cross_raw = None
        self.cross_as = None
        self.cross = None
        self.cross_full_as = None
        self.cross_full = None
        self._process_matrices()

    def __repr__(self):
        """
        Returns a string representation of the CrossMatrixValidator object.

        Returns:
            str: A summary of the detected type and label mapping.
        """
        return f"Type_cross: {self.type_cross}\nMap_labels: {self.map_labels}"

    def _process_matrices(self):
        """
        Processes the input data into different matrix forms.

        Depending on the detected type (raw, cross, or full), the method
        generates standardized forms of the matrix.
        """
        self.map_labels = self._create_map_labels()

        if self.type_cross == "raw":
            self.cross_raw = self.data.copy()
            self._cross_from_raw()
            self._full_from_cross()
        elif self.type_cross == "cross":
            self._remap_labels()
            self._raw_from_cross()
            self._full_from_cross()
        elif self.type_cross == "full":
            self._remap_labels()
            self._cross_from_full()
            self._raw_from_cross()

    def _enter_data(self, data) -> pd.DataFrame:
        """
        Converts the input data into a DataFrame and adjusts it based on
        the scheme.

        Args:
            data: Input data in a list, numpy array, or pandas DataFrame
                  format.

        Returns:
            pd.DataFrame: Adjusted DataFrame with integer values.
        """
        data = copy.deepcopy(data)
        # breakpoint()
        if isinstance(data, (list, tuple, np.ndarray)):
            data = pd.DataFrame(data)
            data.columns = range(data.shape[1])
            data.index = range(data.shape[0])
            self.type_cross = "raw"

        if self.scheme == "reverse":
            data = data.T

        data.columns.name = "predicted"
        data.index.name = "true"
        # breakpoint()

        return data.astype(int)

    def _detect_type(self) -> str:
        """
        Detects the type of the input matrix.

        Returns:
            str: One of 'raw', 'cross', or 'full'.
        """
        if CrossMatrixRecognizer.is_raw(self.data):
            return "raw"
        if CrossMatrixRecognizer.is_full(self.data):
            return "full"
        return "cross"

    def _create_map_labels(self) -> dict:
        """
        Creates a mapping of numerical labels to string labels.

        Returns:
            dict: A dictionary mapping numerical labels to string labels.
        """
        if self.map_labels is not None:
            return self.map_labels

        if self.type_cross == "raw":
            prefix = self.label_prefix
            n = self.data.shape[0]
            label_width = 2 if n > 9 else 1
            return {i: f"{prefix}_{i:0>{label_width}}" for i in range(1, n + 1)
                    }

    def _remap_labels(self):
        """
        Remaps numerical labels in the data to string labels using the mapping.
        """
        data = self.data.copy()

        if self.type_cross == "full":
            data = data.iloc[:-1, :-1]

        data_sq = CrossMatrixValidator._make_matrix_square(data, self.scheme)

        if self.map_labels:
            try:
                data.columns = [self.map_labels[key] for key in data.columns]
                data.index = [self.map_labels[key] for key in data.index]
            except KeyError:
                try:
                    data.columns = [
                        self.map_labels[key] 
                        for key in range(1, data.shape[1] + 1)
                    ]
                    # breakpoint()
                    data.index = [
                            self.map_labels[key]
                            for key in range(1, data.shape[0] + 1)
                            ]
                except KeyError:
                    print("`Warning!`  \n\tUnable to fit class map to data. "
                          "Row and column names remain unchanged.  \n"
                          )
        # breakpoint()
        if self.type_cross == "cross":
            self.cross_as = data
            self.cross = data_sq
        elif self.type_cross == "full":
            self.cross_full_as = self._add_sums_cols_rows(data)
            self.cross_full = self._add_sums_cols_rows(data_sq)
        # breakpoint()

    def _raw_from_cross(self):
        """
        Creates a square numeric version of the raw matrix without
        labels or sums.
        """
        cross_raw = self.cross.copy()
        cross_raw.index = range(len(cross_raw.index))
        cross_raw.columns = range(len(cross_raw.columns))
        cross_raw.columns.name = "predicted"
        cross_raw.index.name = "true"
        self.cross_raw = cross_raw

    def _full_from_cross(self):
        """
        Creates the full cross matrix with sums added to rows and columns.
        """
        self.cross_full_as = self._add_sums_cols_rows(self.cross_as)
        self.cross_full = self._add_sums_cols_rows(self.cross)

    def _cross_from_raw(self):
        """
        Creates a labeled cross matrix from a raw numeric matrix.
        """
        cross = self.cross_raw.copy()
        map_labels = self.map_labels
        names = [map_labels[i] for i in range(1, cross.shape[0] + 1)]
        cross.columns = names
        cross.index = names
        cross.columns.name = "predicted"
        cross.index.name = "true"
        self.cross_as = cross
        self.cross = cross

    def _cross_from_full(self):
        """
        Extracts the cross matrix from a full matrix by removing summary
        rows and columns.
        """
        self.cross_as = self.cross_full_as.iloc[:-1, :-1]
        self.cross = self.cross_full.iloc[:-1, :-1]

    @staticmethod
    def _make_matrix_square(matrix: pd.DataFrame, scheme: str) -> pd.DataFrame:
        """
        Ensures the matrix is square by reindexing rows and columns.

        Args:
            matrix (pd.DataFrame): The input matrix.
            scheme (str): Layout scheme ('normal' or 'reverse').

        Returns:
            pd.DataFrame: A square numeric matrix.
        """
        matrix = matrix.copy()
        if scheme == "normal":
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
            pd.DataFrame: A matrix with summary rows and columns.
        """
        df = df.copy()
        df.loc[:, "sums"] = df.sum(axis=1)
        df.loc["sums", :] = df.sum(axis=0)
        df.columns.name = "predicted"
        df.index.name = "true"
        return df.astype(int)
