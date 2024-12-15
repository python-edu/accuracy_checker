# -*- coding: utf-8-*-

import argparse

# local imports
from acc.src.args_data.args_func import FormatHelp as FH

# --

description = FH(
    """The script checks the quality of classification:
        - creates or uses an existing cross matrix
        - calculates accuracy metrics.

Additional help:

- `accuracy usage help`: displays help about script usage
- `accuracy data help`: displays help about input data
- `accuracy metrics help`: displays information about accuracy metrics
- `accuracy formula help`: help with your own custom calculation formulas 
    """
).txt


def parsuj_argumenty():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=description,
        fromfile_prefix_chars="@",
    )
    # --- argumenty głównego parsera ------------------------------------------
    txt = FH("""Path to data file: '*.csv' or images. If images then:
             - the first one is the reference (true values)
             - the second one is the classification result
             (predicted values).""").txt
    txt = FH("""Paths to data files. You can specify from one to 3 paths
    separated by a space, according to the scheme:
    - path.csv: one path - data saved in the '*.csv' file
    - path.tif: one path - image after classification (it will automatically \
      search for a file with reference data `path_ref.tif`)
    - path.tif path.shp: 2 paths, the first is the image after classification \
       and the second is the reference data (image or vector)\
    - path.tif path.json: 2 paths, the first is the image after \
      classification, the second are the names of classes in json format \
      ({'1': ''wheat', ...}); it will automatically search for the file \
      `path_ref.tif` (references)
    - path.shp path.tif path.json: three paths""").txt
    parser.add_argument('path', nargs='+', type=str, help=txt)

    txt = FH(
        """By default, the script displays the results on the screen. This
        option saves the results to separate csv files:
      - average_acc.csv
      - binary_cross.csv,
      - classic_acc.csv,
      - complex_acc.csv
      - cross_full.csv,
      - simple_acc.csv."""
    ).txt
    parser.add_argument("-s", "--save", help=txt, action="store_true")

    txt = FH("""Name or full path to the directory where the results will be
    saved. Name can be a subpath, e.g. `folder/folder/name`. If the directory
    does not exist it will be created. Only works with the `--save` and/or
    '--report' option.
    By default it creates a new directory inside the directory with the data
    file. The name of the new directory is the name of the data file
    with the 'results' extension, e.g.:
    - cross_full.csv -> cross_full_results.
    """).txt
    parser.add_argument("-o", "--out_dir", type=str, help=txt)

    txt = FH(
        """Generates a report in html: all tables in one html file:
    - raport.html."""
    ).txt
    parser.add_argument("-r", "--report", help=txt, action="store_true")

    txt = FH(
        """Data for creating a report is given in the form key=value. Works \
        only with the argument `-r / --report`!!. Default values:
        - title='Image Classification Accuracy Assessment'
        - description='Research project.'
        """).txt
    default = ["title=Image Classification Accuracy Assessment",
               'description=Research project.',
               'report_file=report.html', 'template_file=report_template.html',
               'template_dir=templates']
    parser.add_argument("--report_data", help=txt, nargs='+', type=list,
                        default=default)

    txt = "Precision - number of decimal places."
    parser.add_argument("-p", "--precision", type=int, help=txt, default=4)

    txt = FH("""Use the option to pack the resulting '.csv' files into a zip
    archive.""").txt
    parser.add_argument("-z", "--zip", action="store_true", help=txt)

    txt = FH("""Enter the file name (without extension) if you want to
    change the default name. The default name is the input file
    name + 'results.zip' suffix, e.g.:
    - cross.csv -> cross_results.zip.""").txt
    parser.add_argument("--zip_name", type=str, help=txt)

    txt = FH("""Enter your own formula to calculate accuracy - a string of \
            characters without spaces, surrounded by quotation marks, e.g. \
            "f1=(2*TP)/(2*TP+FP+FN)". Get help: `accuracy formula \
            help`.""").txt
    parser.add_argument("-f", "--formula", type=str, help=txt)

    txt = FH("Column separator in '*.csv' file (usually ','). Enter it \
            manually if auto-detection doesn't work.").txt
    parser.add_argument("--sep", type=str, help=txt)

    txt = FH("""Use the switch when the input data is in a different
    layout than the default, i.e.:
    - when in the cross matrix the columns are `predicted` values instead \
      of `true`
    - when the binary cross has a vertical layout instead of a horizontal \
    one.""").txt
    parser.add_argument("--reversed", action="store_true", help=txt)

    txt = "Displays additional information while the script is running."
    parser.add_argument("-v", "--verbose", action="store_true", help=txt)

    return parser
