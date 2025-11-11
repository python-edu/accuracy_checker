import textwrap
import pandas as pd
from tabulate import tabulate
import argparse
from typing import Any, Optional
from types import SimpleNamespace


class Verbose:
    """
    A utility class for controlled verbose output during script execution.
    It provides formatted output for various data types like dictionaries,
    DataFrames, and lists. Additionally, it formats script arguments if passed.
    """

    def __init__(self, verbose: bool):
        """
        Initializes the Verbose class.

        Args:
            verbose (bool): A flag indicating whether to enable verbose output.
        """
        self.verbose = verbose

    def __call__(self, data: Any, description: Optional[str] = None) -> None:
        """
        Displays formatted information about the provided data.

        Args:
            data (Any): The data to display. Can be of various types like,
                        DataFrame, dictionary, or list.
            description (Optional[str]): A description to display with the data
                                          If None, uses the name of the object.
        """
        if description is None:
            description = getattr(data, '__name__', str(type(data).__name__))
        description = f" {description}"

        if self.verbose:
            if self._is_args_data(data):
                # Display script arguments
                data = self._print_args(data)
                print(description)
                print(data)

            elif isinstance(data, dict):
                # Format and display dictionary
                formatted_data = self._format_dict(data, width=50)
                print(description)
                print("\n".join(formatted_data))

            elif isinstance(data, pd.DataFrame):
                # Format and display DataFrame
                if self._check_df_len(data) <= 160:
                    print(description)

                    table = tabulate(data,
                                     headers="keys",
                                     showindex=True,
                                     tablefmt="pretty"
                                     )
                    table = "\n".join("   " + line for line in table.splitlines())
                    print(table)

            elif isinstance(data, list):
                # Format and display list
                formatted_data = self._format_list(data)
                print(description)
                print(formatted_data)

            else:
                # For other data types, just print the raw data
                print(description)
                print(f"   {data}")
            print()

    def _format_line(self,
                     line: str,
                     width: int,
                     subsequent_indent: str = " "
                     ) -> str:
        """
        Formats a single line of text.

        Args:
            line (str): The string to format.
            width (int): The width of the formatted line.
            subsequent_indent (str): The string to use for indenting
                         subsequent lines.

        Returns:
            str: A formatted string.
        """
        line = self._key2str(line)
        line = " ".join(line.split())
        res = textwrap.fill(line,
                            width=width,
                            subsequent_indent=subsequent_indent
                            )
        return res

    def _format_dict(self, dc: dict, width: int, char: str = " ") -> list[str]:
        """
        Formats a dictionary for display.

        Args:
            dc (dict): The dictionary to format.
            width (int): The maximum line width for formatting.
            char (str): The character used for padding keys.

        Returns:
            list[str]: A list of formatted strings, one for each key-value
                       pair.
        """
        k = max(len(self._key2str(key)) for key in dc.keys()) + 1
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
            ls (list): A list of strings or other objects.

        Returns:
            str: A formatted string with each list item numbered.
        """
        rr = "\n".join(f"{i: >3}. {str(line)}" for i, line in enumerate(ls, 1))
        return rr

    def _check_df_len(self, df: pd.DataFrame) -> int:
        """
        Checks whether a DataFrame is small enough to be displayed in its
        entirety.

        Args:
            df (pd.DataFrame): A pandas DataFrame.

        Returns:
            int: An integer representing the combined length of column names,
                 used to decide if the DataFrame fits on the screen.
        """
        return len("".join(str(name) for name in df.columns)) + 5

    def _print_args(self, args: Any) -> str:
        """
        Formats and displays script arguments.

        Args:
            args (argparse.Namespace or dict): Script arguments.

        Returns:
            str: A formatted string representing the arguments.
        """
        if not isinstance(args, dict):
            args_dict = vars(args)
        else:
            args_dict = args.copy()

        # Define the correct order for keys
        keys_order = [
            "ROOT", "path", "data_type", "func", "path1", "path2", "path3",
            "save", "out_dir", "out_files", "report", "verbose", "precision",
            "sep", "script_name", "map_labels"
        ]

        formatted_args = []

        max_len = max([len(key) for key in keys_order if key in args_dict])
        max_len += 6

        # Format regular arguments
        for key in keys_order:
            if key in args_dict:
                line = self._format_line(args_dict[key],
                                         width=100,
                                         subsequent_indent=3 * ' '
                                         )
                formatted_args.append(f"{key: >{max_len}}:   {line}")

        # Handle report_data separately
        if "report_data" in args_dict:
            formatted_args.append("\n report_data:")
            for key, val in args_dict["report_data"].items():
                formatted_args.append(
                        (f"{key: >{max_len}}:   "
                         f"{self._format_line(val, width=100)}")
                                      )

        return "\n".join(formatted_args)

    def _key2str(self, key: Any) -> str:
        """
        Converts a key to a string.

        Args:
            key (Any): The key to convert.

        Returns:
            str: A string representation of the key.
        """
        if callable(key):
            key = key.__name__
        elif not isinstance(key, str):
            key = str(key)
        return key.strip()

    def _is_args_data(self, data: Any) -> bool:
        """
        Checks if the data is an instance of argparse.Namespace or
        types.SimpleNamespace

        Args:
            data (Any): The data to check.

        Returns:
            bool: True if the data is an argparse.Namespace instance,
                  else False.
        """
        # return isinstance(data, argparse.Namespace)
        return isinstance(data, (argparse.Namespace, SimpleNamespace))
