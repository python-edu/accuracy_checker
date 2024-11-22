# --- ---*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd

# local imports

# global variables
predict_name = 'predicted'
true_name = 'true'

# ---

"""The module provides classes for:
    - calculating 'cross matrix' from data containing true and predicted
      labels.
    - if verbal names of classes are available, they are used
"""
# ---


class RawData:
    """
    Stores classification results in a table with 2 columns:
    - First column: true values (actual classes)
    - Second column: predicted values (predicted classes)

             |  true_values | predicted |
             | ------------ | --------- |
             |     int      |    int    |
             |     ...      |    ...    |

    Input:
        - Columns must be in order [true_values, predicted]
        - Column names do not matter (data can be without column names)
        - Data can be a list, tuple, numpy array, or pandas DataFrame
    """

    def __init__(self, data, map_labels=None, default_label="cl"):
        """
        Args:
            - data: Input data (list, tuple, numpy array, or DataFrame)
            - map_labels: Dictionary mapping class IDs to class names
            - default_label: Default prefix for class labels (e.g., "cl")
        """
        # Prepare and clean the data
        df = self._prepare_dataframe(data)
        df = self._clean_data(df)

        # Extract true and predicted values
        self.true_values = df.iloc[:, 0].astype(int).tolist()
        self.predicted = df.iloc[:, 1].astype(int).tolist()

        # Create the label map
        self.map_labels = self._generate_map_labels(df,
                                                    map_labels,
                                                    default_label
                                                    )

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
            raise ValueError("Data must be a DataFrame, list, tuple, \
                    or numpy array!")

        # if df.shape[1] != 2:
        if df.shape[1] not in (2, 3):
            raise ValueError("Data must have exactly 2 columns!")
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
        # Convert both columns to numbers
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        # Drop rows with invalid (NaN) values
        return df.dropna()

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
        if map_labels is not None:
            return map_labels
        # if `map_labels` is not given
        #  - if the data contains class descriptions - use them
        if df.shape[1] == 3:
            map_labels = dict(zip(df.iloc[:, 0], df.iloc[:, -1]))
            return map_labels
        else:
            # Collect all unique class IDs from true and predicted values
            all_classes = set(df.iloc[:, 0]).union(df.iloc[:, 1])
            # Create a default label for each class
            map_labels = {cls: f"{default_label}_{cls}" for cls in all_classes}
            return map_labels

    def __repr__(self):
        result = []
        for cls, label in self.map_labels.items():
            result.append(f"  {cls} -> {label}")
        return "\n".join(result)
    # ---


class CrossMatrixRecognizer:
    """
    Utility class to recognize the type of a cross matrix.

    Methods:
        is_raw(df): Checks if the matrix is 'raw' (only numbers, no
                labels or sums).
        is_full(df): Checks if the matrix is 'full' (contains sums of rows
                and columns).
        is_cross(df): Checks if the matrix is 'cross' (contains labels, no sums)
    """
    @staticmethod
    def is_raw(df: pd.DataFrame) -> bool:
        """Check if the DataFrame is in raw cross format."""
        columns_are_numbers = pd.to_numeric(df.columns, errors="coerce").notna()
        index_are_numbers = pd.to_numeric(df.index, errors="coerce").notna()
        result = all(columns_are_numbers) and all(index_are_numbers)
        return result

    @staticmethod
    def is_cross(df: pd.DataFrame) -> bool:
        """Check if the DataFrame is in cross matrix format."""
        is_full = CrossRecognition.is_cross_full(df)
        is_raw = CrossRecognition.is_cross_raw(df)

        # Must be a matrix but neither raw nor full
        result = not is_full and not is_raw
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






class CrossMatrixValidator:
    """
    Validates and processes a cross matrix.

    Attributes:
        data: The original cross matrix (raw, cross, or full).
        type_cross: Detected type of the matrix ('raw', 'cross', or 'full').
        cross_raw: Processed square matrix without labels or sums.
        cross: Matrix with labels.
        cross_full: Matrix with labels and sums.
    """

    def __init__(self,
                 data,
                 map_labels: dict = None,
                 label_prefix="cl",
                 scheme="normal"
                 ):
        """
        Initializes the validator with the input data and optional parameters.

        Args:
            data: The cross matrix, which can be a list of lists, np.array
                  or pd.DataFrame.
            map_labels: Optional mapping for labels.
            label_prefix: Prefix for generating column and row labels
                  e.g. 'cl' -> 'cl_01'.
            scheme: Layout of rows and columns. Options:
                - 'normal': Rows are 'true' values, columns are 'predicted'
                   values.
                - 'reverse': Rows are 'predicted', columns are 'true'.
        """
        self.scheme = scheme
        self.data = self._enter_data(data)
        self.map_labels = map_labels.copy() if map_labels else None
        self.label_prefix = label_prefix

        # Detect matrix type: 'raw', 'cross', 'full'.
        self.type_cross = self._detect_type()

        # Processed matrices.
        self.cross_raw = None
        self.cross_raw_sq = None
        self.cross = None
        self.cross_sq = None
        self.cross_full = None
        self.cross_full_sq = None

        # Process the matrices based on the input.
        self._process_matrices()

    def __repr__(self):
        return f"  Type_cross: {self.type_cross}\n  \
                map_labels: {self.map_labels}\n"

    # --- Main Methods ---

    def _process_matrices(self):
        """
        Processes the input data into different matrix forms.
        """
        self._make_cross_raw()
        self.map_labels = self._create_map_labels()
        self.cross = self._make_cross(self.cross_raw, self.map_labels)
        self.cross_sq = self._make_cross(self.cross_raw_sq, self.map_labels)
        self._make_cross_full()

    def _enter_data(self, data) -> pd.DataFrame:
        """
        Converts the input data into a DataFrame and adjusts it based on
        the scheme.

        Args:
            data: Input data, which can be a list, np.array, or pd.DataFrame.

        Returns:
            A DataFrame with adjusted layout and integer values.
        """
        data = copy.deepcopy(data)

        if isinstance(data, (list, tuple, np.ndarray)):
            data = pd.DataFrame(data)
            data.columns = range(data.shape[1])
            data.index = range(data.shape[0])

        if self.scheme == 'reverse':
            data = data.T

        data.columns.name = "predicted"
        data.index.name = "true"
        return data.astype(int)

    def _detect_type(self) -> str:
        """
        Detects the type of the input matrix.

        Returns:
            A string indicating the type ('raw', 'cross', 'full').
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
            A dictionary mapping numerical labels to string labels.
        """
        if self.map_labels is not None:
            return self.map_labels

        df = self.data.copy()
        if self.type_cross == 'full':
            df = df.iloc[:-1, :-1]

        if self.scheme == 'normal':
            all_classes = list(df.columns)
        else:
            all_classes = list(df.index)

        if self.type_cross in ('full', 'cross'):
            map_labels = {i: name for i, name in enumerate(all_classes, 1)}
        else:
            map_labels = {i: f"{self.label_prefix}_{i}" for i \
                    in range(1, len(all_classes) + 1)}
        return map_labels

    def _make_cross_raw(self):
        """
        Creates a square version of the raw matrix without labels or sums.
        """
        if self.type_cross in ('raw', 'cross'):
            matrix = self.data.copy()
        elif self.type_cross == 'full':
            matrix = self.data.iloc[:-1, :-1]

        matrix_sq = self._make_matrix_square(matrix, self.scheme)

        self.cross_raw_sq = matrix_sq.copy()
        self.cross_raw_sq.index = range(1, matrix_sq.shape[0] + 1)
        self.cross_raw_sq.columns = range(1, matrix_sq.shape[1] + 1)
        self.cross_raw_sq.columns.name = "predicted"
        self.cross_raw_sq.index.name = "true"

        self.cross_raw = self._remove_empty_predicted()

    def _make_cross_full(self):
        """
        Creates the full cross matrix with sums added to rows and columns.
        """
        self.cross_full = self._add_sums_cols_rows(self.cross)
        self.cross_full_sq = self._add_sums_cols_rows(self.cross_sq)

    # --- Static Utility Methods ---

    @staticmethod
    def _make_matrix_square(matrix: pd.DataFrame, scheme: str) -> pd.DataFrame:
        """
        Ensures the matrix is square by reindexing rows and columns.

        Args:
            matrix: The input matrix.
            scheme: Layout scheme ('normal' or 'reverse').

        Returns:
            A square DataFrame.
        """
        matrix = matrix.copy()
        if scheme == 'normal':
            matrix = matrix.reindex(index=matrix.columns, fill_value=0)
        else:
            matrix = matrix.reindex(columns=matrix.index, fill_value=0)
        return matrix

    @staticmethod
    def _make_cross(matrix: pd.DataFrame, map_labels: dict) -> pd.DataFrame:
        """
        Creates a cross matrix with labeled rows and columns.

        Args:
            matrix: Input matrix.
            map_labels: Mapping of numerical labels to string labels.

        Returns:
            A labeled cross matrix.
        """
        cross = matrix.copy()
        row_labels = [map_labels[i] for i in matrix.index]
        col_labels = [map_labels[i] for i in matrix.columns]

        cross.columns = col_labels
        cross.index = row_labels
        cross.columns.name = "predicted"
        cross.index.name = "true"
        return cross

    @staticmethod
    def _add_sums_cols_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds sums to the rows and columns of the matrix.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with sums added.
        """
        df = df.copy()
        df.loc[:, "sums"] = df.sum(axis=1)
        df.loc["sums", :] = df.sum(axis=0)
        df.columns.name = "predicted"
        df.index.name = "true"
        return df.astype(int)

    # --- Helper Methods ---

    def _remove_empty_predicted(self) -> pd.DataFrame:
        """
        Removes rows and columns with only zero values from the raw matrix.

        Returns:
            The cleaned matrix.
        """
        cross = self.cross_raw_sq.copy()
        if self.scheme != 'normal':
            cross = cross.T

        rows_sum = cross.sum(axis=1)
        idx = rows_sum == 0

        if idx.any():
            cross = cross.drop(index=cross.index[idx])

        if self.scheme != 'normal':
            cross = cross.T
        return cross


class CrossMatrixValidator_old:
    """
    Validates and processes a cross matrix.

    Attributes:
        data: Original cross matrix (raw, cross, or full).
        type_cross: Detected type of the matrix ('raw', 'cross', or 'full').
        cross_raw: Processed square matrix without labels or sums.
        cross: Matrix with labels.
        cross_full: Matrix with labels and sums.
    """
    def __init__(self,
                 data,
                 map_labels: dict = None,
                 label_prefix="cl",
                 scheme="normal",
                 ):
        """
        Args:
          - data:  cross matrix, lista list, np.array lub pd.DataFrame
          - label_prefix: str, label_prefix do tworzenia nazw koumn i wierszy
                        np. cl --> cl_01, cl_02, ...

          - scheme: str, określa układ kolumn i wierszy:
                    - normal:  układ w wierszach `true` w kolumnach `predicted`
                    - reverse: układ odwrócony

          - type_cross: str, określa zawartość cross matrix:
                   - raw:  same liczby, bez nazw kolumn i wierszy, bez sum
                   - cross: liczby z opisami kolumn/wierszy, bez sum
                   - full: wszystko, z sumami w wierszach i kolumnach
        """

        self.scheme = scheme
        self.data = self._enter_data(data)

        # detect matrix type: raw, cross, full
        self.type_cross = self._detect_type()
        self.map_labels = map_labels.copy() if map_labels else None
        self.label_prefix = label_prefix

        self.cross_raw = None
        self.cross_raw_sq = None
        self.cross = None
        self.cross_sq = None
        self.cross_full = None
        self.cross_full_sq = None

        self._process_matrices()
    # --- ---

    def __repr__(self):
        return f"""\n\tType_cross: {self.type_cross}
        map_labels: {self.map_labels}\n"""

    def _process_matrices(self):
        """The method performs most of the calculations, so they can be
        triggered by other methods, e.g. when updating the value of an
        attribute.
        """

        # --- completes missing rows and columns (square) and assigns the
        #     result to the appropriate variable (cross_raw, cross, cross_full)

        # --- create `cross_row` (if it doesn't exist)
        self._make_cross_raw()
        self.map_labels = self._create_map_labels()

        self.cross = self._make_cross(self.cross_raw, self.map_labels)
        self.cross_sq = self._make_cross(self.cross_raw_sq, self.map_labels)
        self._make_cross_full()
        # ---

    @staticmethod
    def _make_matrix_square(matrix, scheme):
        matrix = matrix.copy()
        if scheme == 'normal':
            matrix = matrix.reindex(index=matrix.columns, fill_value=0)
        else:
            matrix = matrix.reindex(columns=matrix.index, fill_value=0)
        return matrix

    @staticmethod
    def _make_cross(matrix, map_labels):
        cross = matrix.copy()
        row_labels, col_labels = CrossMatrixValidator._create_cols_rows_names(
                cross,
                map_labels
                )
        cross.columns = col_labels
        cross.index = row_labels
        cross.columns.name = predict_name  # 'predicted'
        cross.index.name = true_name  # 'true'
        return cross
        # ---

    @staticmethod
    def _create_cols_rows_names(matrix, map_labels):
        row_labels = [map_labels[i] for i in matrix.index]
        col_labels = [map_labels[i] for i in matrix.columns]
        return row_labels, col_labels

    def _enter_data(self, data) -> pd.DataFrame:
        data = copy.deepcopy(data)

        if isinstance(data, (list, tuple, np.ndarray)):
            data = pd.DataFrame(data)
            cols = list(range(data.shape[1]))
            rows = list(range(data.shape[0]))
            data.columns = cols[:]
            data.index = rows[:]

        if self.scheme == 'reverse':
            data = data.T
        data.columns.name = predict_name  # 'predicted'
        data.index.name = true_name  # 'true'
        return data.astype('int')
        # ---

    @staticmethod
    def _add_sums_cols_rows(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        sum_row = df.sum(axis=1).to_numpy()
        df.loc[:, "sums"] = sum_row

        sum_col = df.sum(axis=0).to_numpy()
        df.loc["sums", :] = sum_col

        df.columns.name = predict_name  # 'predicted'
        df.index.name = true_name  # 'true'

        return df.astype("int")

    def _detect_type(self):
        """Detects the type of the input matrix."""
        if CrossMatrixRecognizer.is_raw(self.data):
            return "raw"
        if CrossMatrixRecognizer.is_full(self.data):
            return "full"
        return "cross"

    def _create_map_labels(self):
        if self.map_labels is not None:
            return self.map_labels

        df = self.data.copy()
        if self.type_cross == 'full':
            df = df.iloc[:-1, :-1]

        # all_classes = sorted(set(df.columns).union(df.index))
        if self.scheme == 'normal':
            all_classes = list(df.columns)
        else:
            all_classes = list(df.index)

        if self.type_cross in ('full', 'cross'):
            map_labels = {i: name for i, name in enumerate(all_classes, 1)}
        else:
            # map_labels = {i: f"{self.label_prefix}_{i}" for i, _ \
            #         in enumerate(all_classes, 1)}
            map_labels = {i: f"{self.label_prefix}_{i}" for i \
                    in range(1, len(all_classes) + 1)}
        return map_labels

    def _make_cross_raw(self):
        """Creates a square version of the matrix without labels or sums."""
        # matrix: can be (depending on input) numbers only or
        # numbers + column descriptions
        if self.type_cross in ('raw', 'cross'):
            matrix = self.data.copy()
        elif self.type_cross == 'full':
            matrix = self.data.iloc[:-1, :-1]
        # breakpoint()
        matrix_sq = self._make_matrix_square(matrix, self.scheme)

        self.cross_raw_sq = matrix_sq.copy()
        self.cross_raw_sq.index = range(1, matrix_sq.shape[0] + 1)
        self.cross_raw_sq.columns = range(1, matrix_sq.shape[1] + 1)

        # module global variables: true_name, predict_name
        self.cross_raw_sq.columns.name = predict_name
        self.cross_raw_sq.index.name = true_name

        self.cross_raw = self._remove_empty_predicted()
        # module global variables: true_name, predict_name
        self.cross_raw.columns.name = predict_name
        self.cross_raw.index.name = true_name
        # ---

    def _make_cross_full(self):
        self.cross_full = self._add_sums_cols_rows(self.cross)
        self.cross_full_sq = self._add_sums_cols_rows(self.cross_sq)
    

    def _make_cross_square(self) -> None:
        df = self.data.copy()
        if self.type_cross == 'full':
            # the name of the column and row containing summaries can be 
            # different, e.g. `sum_row` and `sum_col`. Sets the same name
            # for both cases, e.g. `sums`
            name = 'sums'
            df.columns = [*df.columns[:-1], name]
            df.index = [*df.index[:-1], name]
            df.columns.name = predict_name  # 'predicted'
            df.index.name = true_name  # 'true'

        existing_labels = set(df.index).union(df.columns)
        existing_labels = sorted(existing_labels)
        
        # make square cross
        df_square = df.reindex(index=existing_labels,
                               columns=existing_labels,
                               fill_value=0)
        
        # save results
        if self.type_cross == 'raw':
            self.cross_raw = df_square
        elif self.type_cross == 'cross':
            self.cross = df_square
        else:
            self.cross_full = df_square

    def _remove_empty_predicted(self):
        """Sometimes there are classes with all zeros, which causes Nan values
        and problems with division by zero. The function removes such rows and
        columns.

        Args:
            - cross_raw: pd.crossstab, no row and column descriptions,
              no summaries.
        """
        # jeśli dana klasa ma same zera to suma w wierszu i kolumnie jest
        # taka sama = 0. Numer wiersza i kolumny jest ten sam - macierz
        # kwadratowa.
        cross = self.cross_raw_sq.copy()

        # normal: remove rows
        if self.scheme != 'normal':
            cross = cross.T

        rows_sum = cross.sum(axis=1) # .to_numpy()
        idx = rows_sum == 0

        if idx.any():
            # rows index - do usunięcia
            ri = cross.index[idx]
            cross.drop(index=ri, inplace=True)

        if self.scheme != 'normal':
            cross = cross.T
        return cross

    def _usun_puste(self):
        """Sometimes there are classes with all zeros, which causes Nan values
        and problems with division by zero. The function removes such rows and
        columns.

        Args:
            - cross_raw: pd.crossstab, no row and column descriptions,
              no summaries.
        """
        # jeśli dana klasa ma same zera to suma w wierszu i kolumnie jest
        # taka sama = 0. Numer wiersza i kolumny jest ten sam - macierz
        # kwadratowa.
        cross = self.cross_raw.copy()
        rows_sum = cross.sum(axis=1).to_numpy()
        idx = rows_sum == 0

        if idx.any():
            # rows index - do usunięcia
            ri = cross.index[idx]

            # cols index - do usunięcia
            ci = cross.columns[idx]

            cross.drop(index=ri, inplace=True)
            cross.drop(columns=ci, inplace=True)

        return cross

    def _number_labels_to_strings(self):
        "When column and row labels are numbers - converts them to strings."
        cross = self.cross_raw.copy()
        kol_name = cross.columns.name
        row_name = cross.index.name

        kols = [str(x) for x in cross.columns]
        rows = [str(x) for x in cross.index]

        cross.columns = kols
        cross.columns.name = kol_name

        cross.index = rows
        cross.index.name = row_name
        return cross.astype("int")

    def _add_text_labels(self):
        """Creates a new cross matrix in which column and row names are
        changed from numbers to verbal descriptions, e.g. oats, rye, etc.
        """
        cross = self.cross_raw.copy()

        # --- pobierz aktualne liczbowe nazwy kolumn i wierszy
        all_names = cross.columns.to_list()

        if "sums" in all_names:
            old = all_names[:-1]
        else:
            old = all_names[:]
        # --- konwersja na string
        old = [str(x) for x in old]

        kols = [self.map_labels.get(key) for key in old]
        rows = kols[:]

        if len(rows) < len(all_names):
            kols.append("sums")
            rows.append("sums")

        cross.columns = kols
        cross.index = rows
        cross.axes[1].name = "predicted"
        cross.axes[0].name = "true"

        return cross.astype("int")


# class ConfusionMatrix:
#     """
#     # Result
#       Returns cross matrix, in 3 three formats:
#       1. 'cross_raw': raw result from `pd.crosstab()`, which:
#             - has changed the names of column one and row one to
#               `predicted` and `true_values`
#             - no summarized rows and columns
#             - no verbal descriptions of classes
#             - equalized number of rows and columns
#             - deleted rows and columns with all zeros
#             - numbers in row and column names changed to `str`
#             - table values changed to `int`
# 
#       1. 'cross_raw': surowy wynik z `pd.crosstab()`, który:
#             - ma zmienione nazwy kolumny pierwszej i wiersza pierwszego na
#               `predicted` i `true_values`
#             - bez podsumowanych wierszy i kolumn
#             - bez opisów słownych klas
#             - wyrównana liczba wierszy i kolumn
#             - usunięte wiersze i kolumny z samymi zerami
#             - liczby w nazwach wierszy i kolumn zamienione na `str`
#             - wartości tabeli zamienione na `int`
# 
#               predicted    1  2  3  4
#               true
#               1            3  0  1  0
#               2            1  3  1  0
#               3            0  2  2  0
#               4            1  1  0  0
# 
# 
#       2. 'cross': to samo co cross_raw ale z dodanymi opisami wierszy i kolumn.
#             Opisy zależą od danych: albo są wzięte z pierwszej kolumny danych
#             albo wygenerowane ze stałej etykiety z kolejnym numerem. Przykład
#             poniżej zawiera domyślną etykietę klas.
# 
#               predicted  cl_1  cl_2  cl_3  cl_4
#               true
#               cl_1          3     0     1     0
#               cl_2          1     3     1     0
#               cl_3          0     2     2     0
#               cl_4          1     1     0     0
# 
#       3. `cross_full`:  pełna macierz z sumami wierszy i kolumn
# 
#               predicted  cl_1  cl_2  cl_3  cl_4  sum_row
#               true
#               cl_1          3     0     1     0        4
#               cl_2          1     3     1     0        5
#               cl_3          0     2     2     0        4
#               cl_4          1     1     0     0        2
#               sum_col       5     6     4     0       15
# 
# 
#     # Dane wejściowe / Input data
#       Wyniki klasyfikacji obrazów w postaci tabeli o 2 lub 3 kolumnach. /
#       Image classification results in the form of a table with 2 or 3 columns.
# 
#       1. Dwie kolumny / two columns:
# 
#              |  true_values | predicted |
#              | ------------ | --------- |
#              |     int      |    int    |
#              |     ...      |    ...    |
# 
#       2. Trzy kolumny / three columns:
# 
#              | labels |  true_values | predicted |
#              | ------ | ------------ | --------- |
#              |   str  |     int      |    int    |
#              |   ...  |     ...      |    ...    |
# 
#              gdzie labels to nazwy klas np. owies, przenica, woda, ...
#     """
# 


class CrossMatrix:
    """
    Generates confusion matrices for classification results.

    Attributes:
        map_labels: Dictionary mapping class IDs to class names.
        true_values: List of true class IDs.
        predicted: List of predicted class IDs.
        cross_raw: Confusion matrix as raw numbers (no labels or summaries).
        cross: Confusion matrix with row and column descriptions.
        cross_full: Confusion matrix with descriptions and summaries.
    """

    def __init__(self, map_labels, true_values, predicted):
        """
        Initializes the CrossMatrix object.

        Args:
            map_labels: Dictionary {class_id: class_name, ...}.
            true_values: List of true class IDs.
            predicted: List of predicted class IDs.
        """
        self.map_labels = map_labels
        self.true_values = true_values
        self.predicted = predicted
        self.cross_raw = None
        self.cross = None
        self.cross_full = None

        self._generate_matrices()

    def _generate_matrices(self):
        """Generates all confusion matrix variants."""
        all_classes = sorted(set(self.true_values) | set(self.predicted))

        # Create raw confusion matrix
        cross_raw = pd.crosstab(
            pd.Series(self.true_values, name="true"),
            pd.Series(self.predicted, name="predicted"),
            dropna=False,
        )
        self.cross_raw = cross_raw.reindex(
            index=all_classes, columns=all_classes, fill_value=0
        )

        # Create labeled matrix
        self.cross = self._add_labels(self.cross_raw)

        # Create labeled matrix with summaries
        self.cross_full = self._add_summaries(self.cross)

    def _add_labels(self, matrix):
        """Adds labels to rows and columns of the confusion matrix."""
        row_labels = [self.map_labels.get(i, f"Unknown_{i}") for i in matrix.index]
        col_labels = [self.map_labels.get(i, f"Unknown_{i}") for i in matrix.columns]
        return matrix.rename(
            index=dict(zip(matrix.index, row_labels)),
            columns=dict(zip(matrix.columns, col_labels)),
        )

    def _add_summaries(self, matrix):
        """Adds summary rows and columns to the confusion matrix."""
        matrix_with_sums = matrix.copy()
        matrix_with_sums.loc["sums"] = matrix_with_sums.sum(axis=0)
        matrix_with_sums["sums"] = matrix_with_sums.sum(axis=1)
        return matrix_with_sums

    def __repr__(self):
        return (
            self.cross_full.to_string(max_rows=5, max_cols=5)
            if self.cross_full is not None
            else "Confusion matrix has not been generated yet."
        )
