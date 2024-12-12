# -*- coding: utf-8-*-

"""Some functions for argument parsers"""

import sys
import json
import re
import textwrap
from pathlib import Path
from collections import Counter

# local import

from acc.src.args_data import help_info as info

# global variable
vector_suffixes = [".shp", ".gpkg"]
imgs_suffixes = [".tiff", ".tif", ".TIF", ".TIFF"]
all_suffixes = imgs_suffixes + vector_suffixes


class FormatHelp:
    """Klasa służy do sformatowania tekstu pomocy. Wywoływana jest, dla
    każdego argumentu indywidualnie."""

    def __init__(self, txt, n=4, width=130):
        # convert to list and strip
        self.txt = [line.strip() for line in txt.splitlines()]
        self.n = n
        self.width = width

        # split list to groups
        self.groups = self._split(self.txt)

        # format groups
        self.groups = self._format_groups(self.groups, n, width)

        # join groups
        self.txt = "\n".join(self.groups)

    def __repr__(self):
        return self.txt

    def _split(self, lines: list) -> dict:
        result = []
        current_group = None

        for line in lines:
            # Jeśli wiersz jest pusty, dodaj go jako pusty wpis
            if line == "":
                result.append({"pusta": ""})
            # Jeśli wiersz zaczyna się od '--', traktuj go jako część podlisty
            elif line.startswith("--"):
                if current_group is None or "podlista" not in current_group:
                    current_group = {"podlista": line}
                    result.append(current_group)
                else:
                    current_group["podlista"] += "\n" + line
            # Jeśli wiersz zaczyna się od '-', traktuj go jako część listy
            elif line.startswith("-"):
                if current_group is None or "lista" not in current_group:
                    current_group = {"lista": line}
                    result.append(current_group)
                else:
                    current_group["lista"] += "\n" + line
            # Jeśli wiersz zaczyna się od '+' lub '|',
            # traktuj go jako część tabeli
            elif line.startswith("+") or line.startswith("|"):
                if current_group is None or "table" not in current_group:
                    current_group = {"table": line}
                    result.append(current_group)
                else:
                    current_group["table"] += "\n" + line
            # Jeśli wiersz zaczyna się od cyfry i kropki lub nawiasu,
            # traktuj go jako punkt numerowany
            elif re.match(r"^\d+[\.\)]", line):
                if current_group is None or "numerowany" not in current_group:
                    current_group = {"numerowany": line}
                    result.append(current_group)
                else:
                    current_group["numerowany"] += "\n" + line
            # Jeśli to jakikolwiek inny wiersz, traktuj go jako opis
            else:
                if current_group is None or "opis" not in current_group:
                    current_group = {"opis": line}
                    result.append(current_group)
                else:
                    current_group["opis"] += " " + line

        return result

    def _format_groups(self, groups: list[dict], n, width):
        res = []
        for gr in groups:
            if "pusta" in gr:
                txt = gr["pusta"]
            elif "lista" in gr:
                txt = self._format_list(gr["lista"], n, width)
            elif "podlista" in gr:
                txt = self._format_sublist(gr["podlista"], n + 2, width)
            elif "numerowany" in gr:
                txt = self._format_numerowany(gr["numerowany"], n, width)
            elif "opis" in gr:
                txt = self._format_opis(gr["opis"], width)
            elif "table" in gr:
                txt = self._format_table(gr["table"], n)

            res.append(txt)
        return res

    def _format_numerowany(self, txt, n, width):
        """Args:
        - txt:  '1. line\n2. line\n3.line..'
        - n:  wielkość wcięcia listy: n * ' '
        - w:  długość maksymalna linii
        """
        # ['1. line', '2. line', ...]
        txt = [line.strip() for line in txt.splitlines()]

        # zamienia kilkukrotne spacje na pojedyncze np. 'abc    xx` -> 'abc xx'
        txt = [" ".join(line.split()) for line in txt]

        # zawija długie linie
        txt = [
            textwrap.fill(line,
                          width=width,
                          subsequent_indent=3 * " ") for line in txt
        ]

        txt = "\n".join(txt)
        return txt

    def _format_list(self, txt, n, width):
        """Args:
        - txt:  '- line\n- line\n-line..'
        - n:  wielkość wcięcia listy: n * ' '
        - w:  długość maksymalna linii
        """
        # ['- line', '- line', ...]
        txt = [line.strip() for line in txt.splitlines()]

        # zamienia kilkukrotne spacje na pojedyncze np. 'abc    xx` -> 'abc xx'
        txt = [" ".join(line.split()) for line in txt]

        # zawija długie linie
        txt = [
            textwrap.fill(line, width=width, subsequent_indent=(2 + n) * " ")
            for line in txt
        ]

        txt = "\n".join(txt)

        # wcina o n*' ' linie
        txt = textwrap.indent(txt,
                              n * " ",
                              predicate=lambda line: line.startswith("-")
                              )
        return txt

    def _format_sublist(self, txt, n, width):
        """Args:
        - txt:  '-- line\n-- line\n--line..'
        - n:  wielkość wcięcia listy: n * ' '
        - w:  długość maksymalna linii
        """
        txt = [line.strip() for line in txt.splitlines()]
        txt = [" ".join(line.split()) for line in txt]
        txt = [
            textwrap.fill(line, width=width, subsequent_indent=(2 + n) * " ")
            for line in txt
        ]

        # wcina o n*' ' linie
        txt = "\n".join(txt)
        txt = textwrap.indent(
            txt, n * " ", predicate=lambda line: line.startswith("--")
        )

        txt = txt.replace("-- ", "")
        txt = txt.replace("--", "")
        return txt

    def _format_opis(self, opis, width):
        opis = [line.strip() for line in opis.splitlines()]
        opis = [" ".join(line.split()) for line in opis]
        opis = " ".join(opis)
        opis = textwrap.fill(opis, width=width, subsequent_indent=(3) * " ")
        return opis

    def _format_table(self, table, n):
        table = [line.strip() for line in table.splitlines()]
        table = [f'{" "*n}{line}' for line in table]
        # table = [' '.join(line.split()) for line in table]
        table = "\n".join(table)
        # table = textwrap.fill(table, width=width)
        return table


def check_file_path(path: str) -> str:
    """
    Validates the existence of a file at the given path.

    Args:
        path (str): Path to the file to validate.

    Returns:
        str: The absolute, resolved path to the file.

    Raises:
        SystemExit: If the path does not point to an existing file.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        msg = f"\n  `{path}` is not a valid file path.\n"
        sys.exit(msg)

    return str(path_obj.resolve())


def check_dir(path_csv: str, out_dir: str = None) -> str:
    """
    Determines or creates an output directory for saving results.

    Rules for output directory:
        - If `out_dir` is None, a directory is created in the same location
          as the `path_csv` file, with the name `<file_name>_results`.
        - If `out_dir` is a name, it is treated as a directory name relative
          to the directory containing the `path_csv` file.
        - If `out_dir` is an absolute path, it is used directly.

    Args:
        path_csv (str): Path to the input CSV file.
        out_dir (str, optional): Name or path for the output directory.

    Returns:
        str: The absolute, resolved path to the output directory.
    """
    path_csv = Path(path_csv).resolve()

    if out_dir is None:
        # Default directory name: <file_name>_results
        name = f"{path_csv.stem}_results"
        out_dir = path_csv.with_name(name)
    else:
        # Convert out_dir to a Path object
        out_dir = Path(out_dir)

        # If out_dir is a relative path, resolve it relative to the CSV file
        if not out_dir.is_absolute():
            out_dir = path_csv.with_name(out_dir.name)

    return str(out_dir.resolve())


def parse_report_data(data: list[str]) -> dict:
    """
    Parses key-value pairs from a list of strings into a dictionary.

    Args:
        data (list[str]): A list of strings, each formatted as "key=value".

    Returns:
        dict: A dictionary where keys and values are extracted from the input.

    Raises:
        ValueError: If any string in the list does not contain
                    an '=' character.
    """
    report_args = dict(item.split("=", 1) for item in data)
    return report_args


def detects_separator(path: str) -> str:
    """
    Automatically detects the column separator in a CSV file.

    Supported separators:
        - Tab (`\\t`)
        - Semicolon (`;`)
        - Comma (`,`)

    Args:
        path (str): Path to the CSV file.

    Returns:
        str: Detected column separator (`\\t`, `;`, or `,`).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no valid separator is detected.
    """
    separators = ["\t", ";", ","]
    try:
        with open(path, "r") as file:
            # Read the first three lines of the file
            lines = [file.readline() for _ in range(3)]
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")

    # Combine lines into a single string and count separator occurrences
    combined_text = "".join(line.strip() for line in lines)
    counts = Counter(combined_text)
    separator_counts = [(sep, counts[sep]) for sep in separators]

    # Return the separator with the highest count
    column_sep = max(separator_counts, key=lambda item: item[1])
    if column_sep[1] == 0:
        raise ValueError(f"No valid separator detected in file: {path}")

    return column_sep[0]


def search_reference_file(path: str) -> str | bool:
    """
    Searches for a reference file related to the given file.

    Supported file extensions:
        - `.tif`, `.tiff`
        - `.shp`, `.gpkg`

    Args:
        path (str): Path to the base file.

    Returns:
        str: Absolute path to the reference file if found.
        bool: False if no reference file is found.
    """
    suffix_list = [".tif", ".tiff", ".TIF", ".TIFF", ".shp", ".gpkg"]
    base_folder = Path(path).resolve().parent
    base_name = Path(path).stem
    potential_files = ([base_folder / f"{base_name}_ref{ext}" for
                        ext in suffix_list])

    # Check if any of the potential reference files exist
    for ref_file in potential_files:
        if ref_file.is_file():
            return str(ref_file.resolve())

    return False


def search_json_file(path: str) -> dict | None:
    """
    Searches for a JSON file in the given directory.

    The JSON file is expected to contain a class map. If found, the first
    matching file is loaded. Keys in the JSON are converted to integers
    if possible.

    Args:
        path (str): Path to the directory to search for a JSON file.

    Returns:
        dict: The class map loaded from the JSON file.
        None: If no valid JSON file is found or parsing fails.

    Raises:
        OSError: If there are issues reading the JSON file.
    """
    path = Path(path).resolve()

    # Find the first JSON file in the directory
    json_file = next(path.glob("*.json"), None)

    if json_file:
        try:
            with open(json_file, "r") as file:
                data = json.load(file)
                # Attempt to convert keys to integers
                try:
                    return {int(key): value for key, value in data.items()}
                except (ValueError, TypeError):
                    return data
        except (json.JSONDecodeError, OSError):
            return None

    return None


def paths_decoder(args):
    """
    Decodes and validates paths provided as arguments.

    The function supports 1 to 3 paths, handling various configurations:
    - Single path:
        - '*.csv' file sets `args.path`.
        - '*.tif' image sets `args.path` and searches for reference data.
    - Two paths:
        - '*.tif' and '*.json': sets `args.path` and `args.path3`, searches
          for reference.
        - '*.tif' and vector file ('*.shp', '*.gpkg'): sets `args.path` and
          `args.path2`.
        - '*.csv' and '*.json': sets `args.path` and `args.path2`.
    - Three paths:
        - Image, vector, and JSON file are expected. Sets `args.path`,
          `args.path2`, `args.path3`.

    Additional options:
    - 'help' with 'usage', 'data' or 'metrics' in paths sets `args.help_usage`,
       `args.help_data` or `args.help_metrics`.

    Args:
        args (Namespace): Argument namespace containing a `path` list.

    Returns:
        Namespace: Updated namespace with `path`, `path2`, and `path3`
                   attributes.

    Raises:
        SystemExit: If invalid paths are provided or an unsupported
                    configuration is detected.
    """
    if len(args.path) > 3:
        msg = f"\n\t{len(args.path)} paths entered - maximum 3 allowed.\n"
        sys.exit(msg)

    paths = args.path[:]

    # Check for help requests
    if "usage" in paths and "help" in paths:
        setattr(args, "help_usage", True)
        return args
    if "data" in paths and "help" in paths:
        setattr(args, "help_data", True)
        return args
    if "metrics" in paths and "help" in paths:
        setattr(args, "help_metrics", True)
        return args

    # Handle three paths
    if len(paths) == 3:
        if Path(paths[-1]).suffix != ".json":
            sys.exit("\n\tThe last path should point to a `.json` file: "
                     f"{paths[-1]}")
        if Path(paths[1]).suffix not in vector_suffixes:
            sys.exit("\n\tThe second path should point to a vector file: "
                     f"{paths[1]}")

        setattr(args, "path", check_file_path(paths[0]))
        setattr(args, "path2", check_file_path(paths[1]))
        setattr(args, "path3", check_file_path(paths[2]))

    # Handle two paths
    elif len(paths) == 2:
        if (Path(paths[0]).suffix in imgs_suffixes and
                Path(paths[-1]).suffix == ".json"):
            args.path = check_file_path(paths[0])
            args.path2 = None
            args.path3 = check_file_path(paths[-1])
        elif (
            Path(paths[0]).suffix in imgs_suffixes
            and Path(paths[-1]).suffix in vector_suffixes
        ):
            args.path = check_file_path(paths[0])
            args.path2 = check_file_path(paths[-1])
            args.path3 = None
        elif (
            Path(paths[0]).suffix in imgs_suffixes
            and Path(paths[-1]).suffix in imgs_suffixes
        ):
            args.path = check_file_path(paths[0])
            args.path2 = check_file_path(paths[-1])
        elif (Path(paths[0]).suffix == ".csv" and
                Path(paths[-1]).suffix == ".json"):
            args.path = check_file_path(paths[0])
            args.path2 = check_file_path(paths[-1])
        else:
            sys.exit("\n\tInvalid input file paths! See help.\n")

    # Handle a single path
    elif len(paths) == 1:
        suffix = Path(paths[0]).suffix
        # breakpoint()
        if suffix == ".csv":
            args.path = check_file_path(paths[0])
        elif suffix in imgs_suffixes:
            args.path = check_file_path(paths[0])
            args.path2 = None
        elif suffix in vector_suffixes:
            msg = (
                f"\n\tOnly one path is entered: {paths[0]}.\n"
                "\tData must be a '*.csv' file or a '*.tif' image.\n"
            )
            sys.exit(msg)
        else:
            sys.exit(
                "\n\tThe provided argument does not meet the script's "
                "requirements.\n"
            )

    return args


def args_validation(args, **kwargs):
    """
    Validates and processes input arguments for a script.

    Key functionalities:
        - Ensures mutually exclusive arguments `save` and `zip` are not
          both set.
        - Decodes paths and handles additional help flags
          (`help_usage`, `help_data`, `help_metrics`).
        - Searches for reference files if required (e.g., for `.tif` images).
        - Creates necessary directories if saving or reporting is enabled.
        - Detects column separator for `.csv` files if not provided.
        - Searches for a JSON file containing class mappings.
        - Configures additional script-specific options.

    Args:
        args (Namespace): Argument namespace containing script options.
        kwargs (dict): Additional keyword arguments (e.g., `script_name`).

    Returns:
        Namespace: Updated namespace with validated and processed arguments.

    Raises:
        SystemExit: If mutually exclusive arguments are set or required files
                    are missing.
    """
    if args.save and args.zip:
        msg = """You can choose whether to save '*.csv' files or '*.zip'
        archives to disk (you can't save both)!!!"""
        msg = " ".join(line.strip() for line in msg.splitlines())
        sys.exit(textwrap.fill(msg, width=120, subsequent_indent="  ") + "\n")

    # Process paths and check additional help flags
    args = paths_decoder(args)

    if (hasattr(args, "help_usage")
        or hasattr(args, "help_data")
        or hasattr(args, "help_metrics")
        ):
        return args

    # Handle single `.tif` files by searching for reference files
    if Path(args.path).suffix in imgs_suffixes and args.path2 is None:
        reference_path = search_reference_file(args.path)
        if not reference_path:
            msg = f"\n\tFor file:\n\t  {args.path}\n"
            msg += "\t  no file with reference values found (*.shp, *.gpkg).\n"
            sys.exit(msg)
        args.path2 = reference_path

    # Set output directory for saving results
    if args.save or args.report or args.zip:
        args.out_dir = check_dir(args.path, args.out_dir)
        if args.save or args.report:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    else:
        delattr(args, "out_dir")

    # Parse report data if reporting is enabled
    if args.report:
        args.report_data = parse_report_data(args.report_data)
        template_dir = Path(args.report_data["template_dir"]).resolve()
        args.report_data["template_dir"] = str(template_dir)
        report_file = Path(args.out_dir) / args.report_data["report_file"]
        args.report_data["report_file"] = str(report_file)
    else:
        del args.report_data

    # Add script-specific options if provided
    if kwargs.get("script_name"):
        args.script_name = kwargs["script_name"]

    # Configure output files if saving is enabled
    if args.save:
        args.out_files = [
            "cross_full",
            "binary_cross",
            "classic_acc",
            "simple_acc",
            "complex_acc",
            "average_acc",
        ]

    # Configure zip file paths
    if args.zip:
        args.zip_name = f"{args.zip_name or Path(args.out_dir).name}.zip"
        args.zip_path = str(
            Path(args.out_dir) / args.zip_name
            if args.save or args.report
            else Path(args.out_dir).with_name(args.zip_name)
        )

    # Detect column separator for `.csv` files
    if args.sep is None and Path(args.path).suffix == ".csv":
        args.sep = detects_separator(args.path)

    # Search for JSON class map file
    root_dir = Path(args.path).parent
    args.map_labels = search_json_file(root_dir)

    return args


def remove_unnecessary_args(args):
    """
    Removes unnecessary attributes from the argument namespace.

    Args:
        args (Namespace): Argument namespace to clean up.

    Returns:
        Namespace: Updated namespace with removed attributes.
    """
    if not args.save:
        delattr(args, "save")

    if not args.zip:
        delattr(args, "zip")
        delattr(args, "zip_name")

    if not args.report:
        delattr(args, "report")

    if args.data_type == "imgs":
        delattr(args, "sep")

    if not args.reversed:
        delattr(args, "reversed")

    return args


def display_additional_help(args):
    """
    Displays additional help information if requested via arguments.

    Args:
        args (Namespace): Argument namespace containing help flags.

    Returns:
        None: Exits the script after displaying help information.
    """
    if (hasattr(args, "help_usage")
        or hasattr(args, "help_data")
        or hasattr(args, "help_metrics")
        ):
        if hasattr(args, "help_data"):
            txt = FormatHelp(info.info_data).txt
        elif hasattr(args, "help_metrics"):
            txt = FormatHelp(info.info_metrics).txt
        elif hasattr(args, "help_usage"):
            txt = FormatHelp(info.info_usage).txt

        print(txt)
        sys.exit()
