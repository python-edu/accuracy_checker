# -*- coding: utf-8-*-

"""
"""
import argparse

# local imports
from acc.src.args_data.args_func import FormatHelp as FH
from acc.src.args_data import args_func as afn
from acc.src.subcommands import from_raw, from_cross_full, from_cross
from acc.src.subcommands import from_cross_raw, from_binary, from_imgs

# --

description = FH(
    """The script checks the quality of classification:
        - creates or uses an existing cross matrix
        - calculates accuracy metrics."""
).txt


info = FH(
        """1. The definitions of the metrics are mainly based on the binary \
        error matrix with the following symbols: TP true positive, TN true \
        negative, FP false positive, FN false negative.

2. Accuracy metrics classically used in remote sensing:
  - OA (overall_accuracy):
   -- OA = sum(TP) / (TP + TN + FP + FN)

  -  PA (producer_accuracy):
   -- PA = TP / (TP + FN)

  -  UA (user_accuracy)
     -- UA = TP / (TP + FP)

  -  OME (omission errors / errors_of_omission):
     -- OME = FN / (TP + FN)

  -  CME (errors_of_commision):
     -- CME = FP / (TP + FP)

  -  NPV (negative predictive value):
     -- NPV = TN/(TN + FN) = 1 − FOR

3. Classification accuracy metrics found in contemporary scientific \
publications (some metrics overlap with some of the metrics mentioned in \
point 1).

These metrics can be conventionally divided into simple metrics \
(calculated directly from the TP, TN, FP and FN values) and complex metrics \
(calculated using simple metrics).

3.1. Simple metrics:

   -   ACC (accuracy):
      -- ACC = (TP+TN)/(P+N) = (TP+TN)/(TP+TN+FP+FN)

   -  PPV (precision or positive predictive value):
      -- PPV = TP / (TP + FP)

   -  PPV (precision or positive predictive):
      -- PPV = TP / (TP + FP)

   -  TPR (sensitivity, recall, hit rate, or true positive rate):
      -- TPR = TP/P = TP/(TP + FN) = 1 − FNR

   -  TNR (specificity, selectivity or true negative rate):
      -- TNR = TN/N = TN/(TN + FP) = 1 − FPR

   -  NPV (negative predictive value):
      -- NPV = TN/(TN + FN) = 1 − FOR

   -  FNR (miss rate or false negative rate):
      -- FNR = FN/P = FN/(FN + TP) = 1 − TPR

   -  FPR (fall-out or false positive rate):
      -- FPR = FP/N = FP/(FP + TN) = 1 − TNR

   -  FDR (false discovery rate):
      -- FDR = FP/(FP + TP) = 1 − PPV

   -  FOR (false omission rate):
      -- FOR = FN/(FN + TN) = 1 − NPV

   -  TS / CSI (Threat score (TS) or critical success index (CSI)):
      -- TS = TP/(TP + FN + FP

   -  MCC (Matthews correlation coefficient):
      -- mcc = (TP*TN - FP*FN) / [(TP+FP) * (TP+FN) * (TN+FP) * (TN+FN)]^0.5

3.2. Complex metrics:
   - PT (Prevalence Threshold):
     -- PT = {[TPR*(1 − TNR)]^0.5 + TNR − 1} / (TPR + TNR − 1)

   - BA (Balanced accuracy):
     -- ba = (TPR + TNR)/2

   - F1 score (is the harmonic mean of precision and sensitivity):
     -- f1 = 2*(PPV*TPR)/(PPV+TPR) = (2*TP)/(2*TP+FP+FN)

   - FM (Fowlkes–Mallows index):
     -- fm = [(TP/(TP+FP))*(TP/(TP+FN))]^0.5 = (PPV * TPR)^0.5

   - BM (informedness or Fowlkes–Mallows index):
     -- bm = TPR + TNR - 1

   - MK (markedness (MK) or deltaP):
     -- mk = PPV + NPV - 1
"""
).txt
# --


def parsuj_argumenty():
    # parser = apk.MyParserWithDefaults(
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
    - path.shp path.tif: 2 paths, the first is the reference data (image or \
      vector) and the second is the image after classification
    - path.tif path.json: 2 paths, the first is the image after \
      classification, the second are the names of classes in json format \
      ({'1': ''wheat', ...}); it will automatically search for the file \
      `path_ref.tif` (references)
    - path.shp path.tif path.json: three paths""").txt
    parser.add_argument('path', nargs='+', type=str, help=txt) 

    # txt = FH("Path to data file: '*.csv' or image").txt
    # parser.add_argument('-p2', '---path2', type=str, help=txt) 

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

    txt = FH("""Displays information (names and calculation formulas) about \
            the calculated statistics.""").txt
    parser.add_argument("-i", "--info", help=txt,
                        action="store_true", default=False)

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
