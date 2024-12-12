import textwrap
import zipfile
import io
import pandas as pd
from pathlib import Path
from tabulate import tabulate

# local import
from acc.src import metrics

# --- dla podpowiadacza:
# breakpoint()
# ---


# def ar2list(ar: np.array) -> list[list]:
#     res = ar.tolist()
#     return res


#def sum_rows_cols(cross: pd.DataFrame):
#    """Oblicza sumy w wierszach i kolumnach dla cross matrix oraz dodaje
#    te sumy do cross matrix jako wiersze i kolumny podsumowujące."""
#    cross = cross.copy()
#    sum_row = cross.sum(axis=1).to_numpy()
#    cross.loc[:, "sum_row"] = sum_row
#
#    sum_kol = cross.sum(axis=0).to_numpy()
#    cross.loc["sum_kol", :] = sum_kol
#
#    cross = cross.astype("int")
#    return cross


# def nazwij_klasy(shape):
#     """Generates class names for a cross matrix when row and column
#     names are not provided. The class names are formatted as
#     'kl_1', 'kl_2', ..., depending on the size of the matrix.
#     
#     Args:
#         shape (tuple): The dimensions of the cross matrix (rows, columns).
#     
#     Returns:
#         list: A list of class names in the format 'kl_1', 'kl_2', ..., 'kl_n'.
#     """
#     n = max(shape)
#     k = 1 if n < 10 else 2
#     names = [f"kl_{i:0>{k}d}" for i in range(1, n + 1)]
#     return names


def acc_from_cross(data, args):
    """
    Computes accuracy metrics from a cross matrix or a binary cross matrix.
    These are standard metrics used in remote sensing.

    Args:
        data: A cross matrix (without row and column summaries) or
              a binary cross matrix.
        args: An object with attributes, usually a namespace object
              from argparse.

    Returns:
        A DataFrame containing the computed classic accuracy metrics.
    """
    if args.data_type == 'binary':
        acc = metrics.AccClasicBin(data, args.precision)
    else:
        acc = metrics.AccClasic(data, args.precision)

    classic_acc = acc.tabela

    return classic_acc


def acc_from_bin_cross(data, args):
    """
    Computes accuracy metrics from a binary cross matrix.
    These metrics are commonly used in machine learning.

    Args:
        data: A binary cross matrix.
        args: An object with attributes, usually a namespace object
              from argparse.

    Returns:
        A tuple containing two DataFrames:
            - The first DataFrame includes metrics like accuracy (acc),
              precision (ppv), sensitivity (tpr), and others.
            - The second DataFrame includes metrics like balanced
              accuracy (ba), F1 score (f1), and related indices.
    """
    acc = metrics.AccIndex(data, precision=args.precision)
    modern1 = {}
    modern2 = {}

    for k, v in vars(acc).items():
        if k in [
            "acc", "ppv", "tpr", "tnr", "npv", "fnr", "fpr",
            "fdr", "foRate", "ts", "mcc",
        ]:
            modern1[k] = v
        elif k in ["pt", "ba", "f1", "fm", "bm", "mk"]:
            modern2[k] = v

    modern1 = pd.DataFrame(modern1)
    modern2 = pd.DataFrame(modern2)

    return modern1, modern2


def format_title(titles: list[str]) -> list[str]:
    """
    Formats a list of titles for better readability in reports.

    Args:
        titles: A list of strings, where each string is a title or a line
                of text to format.

    Returns:
        A list of formatted strings with a width of 90 characters and
        subsequent lines indented by 3 spaces.
    """
    titles = [line.strip() for line in titles]
    titles = [" ".join(line.split()) for line in titles]
    titles = [
        textwrap.fill(
            line,
            width=90,
            subsequent_indent=3 * " "
        ) for line in titles
    ]
    return titles


def save_results(out_dir: str, df_dict: dict) -> list[str]:
    """
    Saves the calculation results to disk as CSV files.

    Args:
        out_dir: A string representing the directory path where the files
                 will be saved.
        df_dict: A dictionary where keys are names of data tables and values
                 are pandas DataFrames. The file name will be derived from
                 the key with a '.csv' extension.

    Returns:
        A list of file paths where the DataFrames were saved.
    """
    recorded = []
    for name, df in df_dict.items():
        name = f"{name}.csv"
        out_path = str(Path(out_dir) / name)
        df.to_csv(out_path, index_label=False)
        recorded.append(out_path)
    return recorded


def zip_results(zip_path: str, df_dict: dict) -> None:
    """
    Saves calculation results directly to a ZIP archive without creating
    intermediate CSV files on disk.

    Args:
        zip_path: A string representing the path to the ZIP archive that
                  will be created.
        df_dict: A dictionary where keys are the names of the data tables
                 and values are pandas DataFrames. The keys will be used as
                 file names in the archive, with a '.csv' extension.

    Returns:
        None
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for name, df in df_dict.items():
            name = f"{name}.csv"
            # Create CSV content in memory
            with io.StringIO() as csv_buffer:
                df.to_csv(csv_buffer)
                csv_buffer.seek(0)
                # Add the CSV file to the ZIP archive
                zipf.writestr(name, csv_buffer.getvalue())


class Verbose:
    """
    A utility class for controlled verbose output during script execution.
    It provides formatted output for various data types like dictionaries,
    DataFrames, and lists.
    """

    def __init__(self, verbose: bool):
        """
        Initializes the Verbose class.

        Args:
            verbose: A boolean indicating whether to enable verbose output.
        """
        self.verbose = verbose

    def __call__(self, data, description=None, args_data=False):
        """
        Displays formatted information about the provided data.

        Args:
            data: The data to display. Can be of various types like,
                  DataFrame, or list.
            description: An optional string to describe the data.
            args_data: A boolean indicating whether the data is script
                       arguments (e.g., from argparse).
        """
        if description is not None:
            description = f"\n\n  ***  {description}  ***\n"

        if self.verbose and args_data:
            # Display script arguments
            print(description)
            data = self._print_args(data)
            print(data)

        elif self.verbose and not args_data:
            print(description)
            if isinstance(data, dict):
                data = self._format_dict(data, width=50)
                data = "\n".join(data)
                print(data)
            elif (isinstance(data, pd.DataFrame)
                  and self._check_df_len(data) <= 160
                  ):
                print(tabulate(data,
                               headers="keys",
                               showindex=True,
                               tablefmt="pretty")
                      )
            elif isinstance(data, list):
                data = self._format_list(data)
                print(data)
            else:
                print(data)
            print()

    def _format_line(self, line: str, **kwargs) -> str:
        """
        Formats a single line of text.

        Args:
            line: The string to format.
            **kwargs: Additional arguments passed to textwrap.fill.

        Returns:
            A formatted string.
        """
        line = self._key2str(line)
        line = " ".join(line.split())
        return textwrap.fill(line, **kwargs)

    def _format_dict(self, dc: dict, width: int, char=" ") -> list[str]:
        """
        Formats a dictionary for display.

        Args:
            dc: The dictionary to format.
            width: The maximum line width for formatting.
            char: The character used for padding keys.

        Returns:
            A list of formatted strings, one for each key-value pair.
        """
        # Calculate padding for keys
        k = max(len(self._key2str(key)) for key in dc.keys()) + 1

        # Format each key-value pair
        res = []
        for key, val in dc.items():
            key = self._key2str(key)
            val = self._key2str(val)
            val = " ".join(val.strip().split())
            line = f"{key:{char}>{k}}:  {val}"
            line = textwrap.wrap(line,
                                 width=width,
                                 subsequent_indent=(k + 6) * " "
                                 )
            res.append("\n".join(line))
        return res

    def _format_list(self, ls: list) -> str:
        """
        Formats a list for display.

        Args:
            ls: A list of strings or other objects.

        Returns:
            A single string with each list item numbered and formatted.
        """
        res = "\n".join(f"{i: >3}. {str(line)}"
                        for i, line in enumerate(ls, 1)
                        )
        return res

    def _check_df_len(self, df) -> int:
        """
        Checks whether a DataFrame is small enough to be displayed in its
        entirety.

        Args:
            df: A pandas DataFrame.

        Returns:
            An integer representing the combined length of column names,
            used to decide if the DataFrame fits on the screen.
        """
        return len("".join(str(name) for name in df.columns)) + 5

    def _print_args(self, args):
        """
        Formats and displays script arguments.

        Args:
            args: An argparse.Namespace object or dictionary of script
                  arguments.

        Returns:
            A formatted string representing the arguments.
        """
        if not isinstance(args, dict):
            args_dict = vars(args)
        else:
            args_dict = args.copy()

        # Setup key order
        keys_order = ["path", "data_type", "func"]
        if hasattr(args, 'path2'):
            keys_order.insert(1, 'path2')
        if hasattr(args, 'path3'):
            keys_order.insert(2, 'path3')
        if hasattr(args, 'save') and args.save:
            keys_order.extend(["save", "out_dir", "out_files"])
        if hasattr(args, 'report') and args.report:
            keys_order.extend(["report", "report_data"])

        # Convert keys to strings
        new_args = {self._key2str(key): val for key, val in args_dict.items()}

        # Add missing keys to the order
        for key in sorted(new_args.keys()):
            if key not in keys_order:
                keys_order.append(key)

        # Format arguments
        res = []
        for key in keys_order:
            line = args_dict[key]
            line = self._format_line(line,
                                     width=110,
                                     subsequent_indent=3 * " "
                                     )
            res.append(f"{key: >13}:   {line}")
        return "\n".join(res)

    def _key2str(self, key) -> str:
        """
        Converts a key to a string.

        Args:
            key: The key to convert.

        Returns:
            A string representation of the key.
        """
        if callable(key):
            key = key.__name__
        elif not isinstance(key, str):
            key = str(key)
        return key.strip()
