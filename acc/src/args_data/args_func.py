# -*- coding: utf-8-*-

"""Some functions for argument parsers"""

import sys
import json
import re
import argparse
import textwrap
from pathlib import Path
from collections import Counter

# local import

from acc.src.args_data import help_info as info


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

    # ---

    def __repr__(self):
        return self.txt
    # ---

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

    # ---

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

    # ---

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

    # ---

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
                              predicate=lambda line: line.startswith("-"))
        return txt

    # ---

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

    # ---

    def _format_opis(self, opis, width):
        opis = [line.strip() for line in opis.splitlines()]
        opis = [" ".join(line.split()) for line in opis]
        opis = " ".join(opis)
        opis = textwrap.fill(opis, width=width, subsequent_indent=(3) * " ")
        return opis

    # ---

    def _format_table(self, table, n):
        table = [line.strip() for line in table.splitlines()]
        table = [f'{" "*n}{line}' for line in table]
        # table = [' '.join(line.split()) for line in table]
        table = "\n".join(table)
        # table = textwrap.fill(table, width=width)
        return table




def check_file_path(path: str) -> Path:
    if not Path(path).is_file():
        # raise argparse.ArgumentTypeError(f"`{path}` is not a valid file path.")
        msg = f"\n  `{path}` is not a valid file path.\n"
        sys.exit(msg)
    else:
        return str(Path(path).resolve())




def check_dir(path_csv: str, out_dir: str) -> str:
    """Building a directory for saving results:
        - by default it creates a new directory inside the directory with
          the data file. The name of the new directory is the name of the
          data file with the 'results' extension, e.g.
          cross_full.csv -> cross_full_results
        - if a directory name is given, it creates a new directory inside
          the data directory
        - if a directory path is given, it creates a new directory from
          this path

    Args:
     - path_csv: str, path to input data file '*.csv'
     - out_dir:  str, nazwa lub path katalogu wyjściowego.
    """
    path_csv = Path(path_csv).resolve()
    # breakpoint()
    if out_dir is None:
        name = f"{path_csv.stem}_results"
        out_dir = path_csv.with_name(name)

    out_dir = Path(out_dir)
    # użytkownik wpisał nazwę
    if not out_dir.is_absolute():
        out_dir = path_csv.with_name(out_dir.name)

    out_dir = out_dir.resolve()
    return str(out_dir)


def parse_report_data(data: list) -> dict:
    """Args:
    - data: ['key=val', 'key=val' ...]
    """
    report_args = dict([item.split("=") for item in data])
    return report_args


def detects_separator(path: str):
    """Automatically detects column separator and decimal separator in
    csv file.
    Args:
        - path: str, path to '*.csv' file (crossmatrix)
    Returns:
        - column_sep: str, ',' or ';' or '\t'
    """
    separators = ['\t', ';', ',']
    with open(path) as f:
        test = [f.readline() for _ in range(3)]
    test = [line.replace('\n', '') for line in test]
    test = ''.join(test)
    counts_all = Counter(test)
    counts_sep = [(key, counts_all[key]) for key in separators]
    column_sep = max(counts_sep, key=lambda item: item[1])
    return column_sep[0]


def search_reference_file(path):
    suffix_list = ['.tif', '.tiff', '.TIF', '.TIFF', '.shp', '.gpkg']
    folder = Path(path).resolve().parent
    name = Path(path).stem
    items = [f'{name}_ref{suffix}' for suffix in suffix_list]
    
    for it in items:
        tmp = folder / it
        if tmp.is_file():
            return str(tmp.resolve())
    return False


def search_json_file(path):
    """Searches the data directory for a '*.json' file, which if present
    contains a class map. The file can have any name and should only be one
    - the first one found will be read."""
    path = Path(path).resolve()
    path = next((path.glob('*.json')), False)
    if path:
        with open(path) as f:
            map_labels = json.load(f)
            # converting keys from str to int type
            try:
                map_labels = {int(key): val for key, val in map_labels.items()}
            except Exception:
                pass
    else:
        map_labels = None
    return map_labels
    # ---


def paths_decoder(args):
    """The path argument is a list of paths from 1 to 3. The function
    supports various path configurations, e.g.:
     - ['*.csv'], [image], [reference, image], ,
     - [reference, image, json_data], [image, json_data].

    The function sets one to three paths:
     - args.path
     - args.path2
     - args.path3

    This additional argument captures values that cause additional
    information (help) to be displayed instead of running the script.

    The args.path setting is important for further processing. If:
    - only args.path exists: this means that csv or images data is entered
    - args.path exists and args.path2 is None: path is the path to the image
      file after classification, the automat will try to find a matching
      reference file
    - args.path exists and args.path2 is not None: both paths for image
      processing are present: the first is the reference, the second is the
      image after classification
    - args.path exists, args.path2 is None and args.path3 is a json file:
      args.path is the path to the image after classification and the automat
      should search for reference data
    - args.path exists and args.path3 exist: csv or image + json (classes)
    - other configurations return an error and end the script
    """
    imgs_suffixes = ['.tiff', '.tif', '.TIF', '.TIFF']
    all_suffixes = ['.tiff', '.tif', '.TIF', '.TIFF', '.shp', '.gpkg']

    if len(args.path) > 3:
        msg = f"\n\t{len(args.path)} paths entered - maximum 3 allowed.\n"
        sys.exit(msg)

    paths = args.path[:]

    # first check if there is a request to display additional help!!!
    if 'data' in paths and 'help' in paths:
        setattr(args, 'help_data', True)
        return args
    if 'metrics' in paths and 'help' in paths:
        setattr(args, 'help_metrics', True)
        return args

    # the path contains 3 files
    if len(paths) == 3:
        if Path(paths[-1]).suffix != '.json':
            msg = f"\n\tThe last path should point to the `.json` file, \
                    and it points to: {paths[-1]}\n"
            sys.exit(msg)

        for i, pt in enumerate(paths, 1):
            # pt = str(Path(pt).resolve())
            pt = check_file_path(pt)
            if i == 1:
                setattr(args, "path", pt)
            else:
                setattr(args, f"path{i}", pt)
    # there are two paths
    elif len(paths) == 2:
        if Path(paths[0]).suffix in imgs_suffixes \
                and Path(paths[-1]).suffix == '.json':
            args.path = check_file_path(paths[0])
            args.path2 = None
            args.path3 = check_file_path(paths[-1])

        elif Path(paths[0]).suffix in all_suffixes \
                and Path(paths[-1]).suffix in imgs_suffixes:
            args.path = check_file_path(paths[0])
            args.path2 = check_file_path(paths[-1])

        elif Path(paths[0]).suffix == '.csv' \
                and Path(paths[-1]).suffix == '.json':
            args.path = check_file_path(paths[0])
            args.path2 = check_file_path(paths[-1])
        else:
            sys.exit("\n\tInvalid input file paths! See help.\n")

    # jeśli jest jeden plik
    elif len(paths) == 1:
        if Path(paths[0]).suffix == '.csv':
            args.path = check_file_path(paths[0])
        elif Path(paths[0]).suffix in ['.tiff', '.tif', '.TIF', '.TIFF']:
            args.path = check_file_path(paths[0])
            args.path2 = None
            
        elif Path(paths[0]).suffix in ['.shp', '.gpkg']:
            m1 = f"\n\tOnly one path is entered: {paths[0]}."
            m2 = "\tIn this case the data must be either a '*.csv' file or"
            m3 = "a '*.tif' image.\n"
            msg = f"{m1}\n{m2} {m3}"
            sys.exit(msg)
    return args


def args_validation(args, **kwargs):
    imgs_suffixes = ['.tiff', '.tif', '.TIF', '.TIFF']
    all_suffixes = ['.tiff', '.tif', '.TIF', '.TIFF', '.shp', '.gpkg']

    # if args.info:
    #     print(kwargs.get('info', 'no information'))
    #     sys.exit(' ')
    # else:
    #     delattr(args, 'info')

    # Checks if `save` and `zip` are not specified at the same time: only
    # one of the options is allowed at a time
    if args.save and args.zip:
        msg = """You can choose whether to save '*.csv' files or
        '*.zip' archives to disk (you can't save both)!!!"""
        msg = " ".join(line.strip() for line in msg.splitlines())
        msg = textwrap.fill(msg, width=120, subsequent_indent='  ')
        sys.exit(msg+'\n')

    # read and sort the paths or additional help
    args = paths_decoder(args)

    if hasattr(args, 'help_data') or hasattr(args, 'help_metrics'):
        return args

    # only the path to the image was entered
    if Path(args.path).suffix in imgs_suffixes and args.path2 is None:
        path = search_reference_file(args.path)
        args.path2 = args.path
        args.path = path

    root_dir = Path(args.path).parent

    # creates a directory to save the results, but only if something needs
    # to be saved
    if args.save or args.report or args.zip:
        args.out_dir = check_dir(args.path, args.out_dir)
        if args.save or args.report:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    else:
        delattr(args, 'out_dir')

    if args.report:
        args.report_data = parse_report_data(args.report_data)

        template_dir = Path(args.report_data['template_dir']).resolve()
        args.report_data["template_dir"] = str(template_dir)

        report_file = args.report_data["report_file"]
        report_file = str(Path(args.out_dir) / report_file)
        args.report_data["report_file"] = report_file
    else:
        del args.report_data
        # breakpoint()

    if kwargs["script_name"]:
        args.script_name = kwargs["script_name"]

    if args.save:
        # args.out_dir = check_dir(args.out_dir, args.path)
        args.out_files = [
            "cross_full",
            "binary_cross",
            "classic_acc",
            "simple_acc",
            "complex_acc",
            "average_acc",
        ]

    if args.zip:
        if args.zip_name is not None:
            args.zip_name = f"{args.zip_name}.zip"
            # args.zip_path = str(Path(args.out_dir).with_name(args.zip_name))
        else:
            args.zip_name = f"{Path(args.out_dir).name}.zip"

        # if it saves data or report then zip puts in the same directory
        if args.save or args.report:
            args.zip_path = str(Path(args.out_dir) / args.zip_name)
        else:
            args.zip_path = str(Path(args.out_dir).with_name(args.zip_name))

    # detecting column separator in '*.csv' file
    suffix = Path(args.path).suffix[1:]  # eg. '.csv' -> 'csv'
    # suffix_list = ['tif', 'tiff', 'TIF', 'TIFF', 'gpkg', 'shp']
    if args.sep is None and suffix == 'csv':
        args.sep = detects_separator(args.path)

    # searching for the file '*.json' which contains the class map
    args.map_labels = search_json_file(root_dir)
    return args


def remove_unnecessary_args(args): 
    if not args.save:
        delattr(args, 'save')

    if not args.zip:
        delattr(args, 'zip')
        delattr(args, 'zip_name')

    if not args.report:
        delattr(args, 'report')

    if args.data_type == 'imgs':
        delattr(args, 'sep')

    if not args.reversed:
        delattr(args, 'reversed')

    if hasattr(args, 'help_data') or hasattr(args, 'help_metrics'):
        keys = ('help_data', 'help_metrics')
        att = {key: getattr(args, key) for key in keys if hasattr(args, key)}

        for key in vars(args).keys():
            delattr(args, key)

        for key, val in att.items():
            setattr(args, key, val)

    return args


def display_additional_help(args):
    if hasattr(args, 'help_data') or hasattr(args, 'help_metrics'):
        if hasattr(args, 'help_data'):
            txt = FormatHelp(info.info_data).txt
        elif hasattr(args, 'help_metrics'):
            txt = FormatHelp(info.info_metrics).txt
        
        print(txt)
        sys.exit()

    return





