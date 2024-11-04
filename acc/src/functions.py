import textwrap
import zipfile
import io
import numpy as np
import pandas as pd
from pathlib import Path

from tabulate import tabulate

from acc.src import metrics

# --- dla podpowiadacza:
# breakpoint()
# ---


def ar2list(ar: np.array) -> list[list]:
    res = ar.tolist()
    return res


# --


def df2list(df: pd.DataFrame) -> list[list]:
    df = df.copy()
    if df.index.name is None and df.columns.name is None:
        name = "-"

    elif df.index.name is not None and df.columns.name is not None:
        name = f"{df.index.name}/{df.columns.name}"
    elif df.index.name is None:
        name = df.columns.name
    else:
        name = df.index.name

    df.index.name = name

    df = df.reset_index()
    cols = df.columns.to_list()
    res = df.to_numpy()
    res = ar2list(res)
    res.insert(0, cols)
    return res


# --


def sum_rows_cols(cross: pd.DataFrame):
    """Oblicza sumy w wierszach i kolumnach dla cross matrix oraz dodaje
    te sumy do cross matrix jako wiersze i kolumny podsumowujące."""
    cross = cross.copy()
    sum_row = cross.sum(axis=1).to_numpy()
    cross.loc[:, "sum_row"] = sum_row

    sum_kol = cross.sum(axis=0).to_numpy()
    cross.loc["sum_kol", :] = sum_kol

    cross = cross.astype("int")
    return cross


# --


def nazwij_klasy(shape):
    """
    Jeśli cross nie ma nazw wierszy i kolumn (cross_raw) to tworzy nazwy klas:
    - kl_01, kl_02,...
    """
    n = max(shape)
    names = [f"kl_{i:0>2}" for i in range(1, n + 1)]
    return names


# ---


def acc_from_cross(data, args):
    """
    Oblicza wskaźniki dokładności na podstawie crossmatrix lub binary cross.
    Oblicza tradycyjne dla teledetekcji wskaźniki.
    Args:
      - data:  - cross matrix (cross), bez podsumowań wierszy i kolumn!!!
               - binary cross matrix (bin_cross)
      - args:  obiekt z atrybutami, zwykle namespase z argparse
    """
    if args.data_type in ["data", "cross", "raw", "full"]:
        acc = metrics.AccClasic(data, args.precision)

    else:
        acc = metrics.AccClasicBin(data, args.precision)

    classic_acc = acc.tabela

    return classic_acc


# ---


def acc_from_bin_cross(data, args):
    """
    Oblicza wskaźniki dokładności na podstawie binary cros.
    Oblicza wskaźniki stosowane w maszynowym uczeniu.
    Args:
      - data:  binary cros matrix (bin_cros)
      - args:  obiekt z atrybutami, zwykle namespase z argparse
    """

    acc = metrics.AccIndex(data, precision=args.precision)
    modern1 = {}
    modern2 = {}

    for k, v in vars(acc).items():
        if k in [
            "acc",
            "ppv",
            "tpr",
            "tnr",
            "npv",
            "fnr",
            "fpr",
            "fdr",
            "foRate",
            "ts",
            "mcc",
        ]:
            modern1[k] = v
        elif k in ["pt", "ba", "f1", "fm", "bm", "mk"]:
            modern2[k] = v

    modern1 = pd.DataFrame(modern1)
    modern2 = pd.DataFrame(modern2)

    return modern1, modern2


# ---


def format_title(titles: list[str]) -> list[str]:
    titles = [line.strip() for line in titles]
    titles = [" ".join(line.split()) for line in titles]
    titles = [
        textwrap.fill(line,
                      width=90,
                      subsequent_indent=3 * " ") for line in titles
    ]
    return titles


# ---


def save_results(out_dir: str, df_dict: dict) -> list[str]:
    """Saves calculation results to disk.
    Args:
        - out_dir: str, path to directory for saving results
        - df_dict: {'name_df' : df, 'name_df': df, ...}, where
          'name_df' is the name of the data table and the file name (after
          adding the 'csv' extension)
    """

    recorded = []

    for name, df in df_dict.items():
        name = f"{name}.csv"
        out_path = str(Path(out_dir) / name)
        df.to_csv(out_path)  # , sep=args.sep)
        recorded.append(out_path)

    return recorded


# ---


def zip_results(zip_path: str, df_dict: dict) -> None:
    """Saves calculation results directly to a 'zip' archive. Does
    not create '*.csv' files on disk. The result is just
    an archive.
    Args:
        - zip_path: str, path to the '*.zip' archive that will be created
        - df_dict: {'name_df' : df, 'name_df': df, ...}
    """

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for name, df in df_dict.items():
            name = f"{name}.csv"
            # creates csv file in ram
            with io.StringIO() as csv_buffer:
                df.to_csv(csv_buffer)  # , index=False)
                csv_buffer.seek(0)
                # Dodanie pliku CSV do archiwum ZIP
                zipf.writestr(name, csv_buffer.getvalue())


# ---


class Verbose:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    # ---

    def __call__(self, data, opis=None, args_data=False):
        if opis is not None:
            opis = f"\n\n  ***  {opis}  ***\n"

        if self.verbose and args_data:
            # only for script arguments
            print(opis)
            data = self._print_args(data)
            print(data)

        elif self.verbose and not args_data:
            print(opis)
            if isinstance(data, dict):
                data = self._format_dict(data, width=50)
                data = "\n".join(data)
                print(data)
            elif isinstance(data, pd.DataFrame) and self._check_df_len(data) <= 160:
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

    # ---

    def _format_line(self, line: str, **kwargs) -> str:
        line = self._key2str(line)
        line = " ".join(line.split())
        line = textwrap.fill(line, **kwargs)
        return line

    # ---

    def _format_dict(self, dc: dict, width: int, char=" ") -> str:
        # fix lengh k: 2 space + max(len(key))
        k = 0
        for key in dc.keys():
            key = self._key2str(key)
            if len(key) > k:
                k = len(key)
        k += 1

        # format lines
        res = []
        for key, val in dc.items():
            key = self._key2str(key)
            val = self._key2str(val)
            val = " ".join(val.strip().split())
            line = f"{key:{char}>{k}}:  {val}"
            line = textwrap.wrap(
                line,
                # initial_indent=n1*' ',
                width=width,
                subsequent_indent=(k + 6) * " ",
            )
            line = "\n".join(line)
            res.append(line)
        # res = '\n'.join(res)
        return res

    # ---

    def _format_list(self, ls: list) -> str:
        lines = [f"{i: >3}. {str(line)}" for i, line in enumerate(ls, 1)]
        return "\n".join(lines)

    # ---

    def _format_subargs(self, dc: dict):
        lines = self._format_dict(dc, width=130, char=".")
        line1 = f"{lines.pop(0): >16}"
        lines = [f"{' '*17}{line}" for line in lines]
        lines.insert(0, line1)
        lines = "\n".join(lines)
        return lines

    # ---

    def _check_df_len(self, df):
        """Checks whether the length of 'pd.DataFrame' allows it to be
        displayed using 'tabulate' - whether it fits on the screen."""
        df_len = "".join([str(name) for name in df.columns])
        df_len = len(df_len) + 5
        return df_len

    # ---

    def _print_args(self, args):
        """
        Args:
            - args:  argparse.Namespace or dict
        """
        if not isinstance(args, dict):
            args_dict = vars(args)
        else:
            args_dict = args.copy()

        # --- setup keys order
        keys_order = ["subcommand", "path"]

        # if args.get('save', False):
        if args.save:
            keys_order.extend(["save", "out_dir", "out_files"])

        # if args.get('report', False):
        if args.report:
            keys_order.extend(["report", "report_data"])

        # --- change key to string key
        new_args = {}
        for key, val in args_dict.items():
            key = self._key2str(key)
            new_args[key] = val

        # --- update `keys_order`
        for key in sorted(new_args.keys()):
            if key not in keys_order:
                keys_order.append(key)

        # --- format args
        res = []
        for key in keys_order:
            line = args_dict[key]
            if key == "report_data" and args.report:
                line = self._format_subargs(line)
                # line = '\n'.join(line)

            elif key == "save" and args.save:
                line = {
                    "out_dir": args_dict.get("out_dir"),
                    "out_files": args_dict.get("out_files"),
                }
                line = self._format_subargs(line)

            else:
                line = self._format_line(
                    line, **{"width": 90, "subsequent_indent": 3 * " "}
                )
            res.append(f"{key: >13}:   {line}")
        return "\n".join(res)

    # ---

    def _key2str(self, key) -> str:
        if callable(key):
            key = key.__name__
        elif not isinstance(key, str):
            key = str(key)
        key = key.strip()
        return key
