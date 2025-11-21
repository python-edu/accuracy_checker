import textwrap
import zipfile
import io
import pandas as pd
from pathlib import Path

# local import
from acc.src.calculations import metrics

# --- dla podpowiadacza:
# breakpoint()
# ---


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
        # ---
