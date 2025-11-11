import re
import textwrap
import json
import base64
from collections import OrderedDict
from pathlib import Path


imgs_replacement = {
        'run1': '-:$ accuracy data2cols.csv',
        'run2': '-:$ accuracy data2cols.csv class_map.json',
        'run3': '-:$ accuracy data3cols.csv',
        'run4': '-:$ accuracy data3cols.csv class_map.json',
        'cross_raw1': '-:$ accuracy cross_raw.csw',
        'cross_raw2': '-:$ accuracy cross_raw.csw class_map.json',
        'cross1': '-:$ accuracy cross.csv',
        'raster1': ('-:$ accuracy raster.tif  # this will generate an error: '
                    'no file `raster_ref`!!!'),
        'raster2': '-:$ accuracy raster.tif  # only if exists `raster_ref.tif`',
        'raster3': ('-:$ accuracy raster.tif class_map.json  # only if exists '
                    '`raster_ref.tif`'),

        }

# info_usage = """
# 1. Launching the program
#     Start the program from an open terminal window in two ways:
# 
#     - `accuracy_gui`- starts a simple GUI (may not expose the full functionality)
#     - `accuracy` - runs the console (CLI) tool (provides full functionality)
# 
#     **Note**:
#     > If installed via the installer, `accuracy_gui` can be started from
#     anywhere without manually activating a virtual environment.
# 
# 2. Required arguments (inputs)
#     The program calculates accuracy on data stored in files.
#     Paths to these files are positional (required) arguments.
#     You can provide inputs in the following ways.
# 
#     2.1. The input is a single `*.csv` file:
#         - raw data: `data2cols.csv` or `data3cols.csv`
#         - confusion matrix: `cross_raw.csv`, `cross.csv` or `cross_full.csv`
#         - binary confusion matrix: `binary_cross.csv`
# 
#         Examples:
#         - `accuracy cross.csv`
#         - `accuracy cross_raw.csv`
#         - `accuracy data3cols.csv`
# 
#     2.2. The input is 2 raster images, in this order:
#         - classified image - usually `*.tif`
#         - reference raster (image/mask) — `*.tif`
#     
#         Example:
#         - `accuracy raster.tif reference_raster.tif`
#     
#     2.3. The input is 2 files, in this order:
#         - classified image - usually `*.tif`
#         - reference vector data - usually `*.shp` or `*.gpkg`
#     
#         Examples:
#         - `accuracy raster.tif reference_vector.shp`
#         - `accuracy raster.tif reference_vector.gpkg`
#     
#     2.4. Optional class map (`*.json`)
#       For any of the cases above, you may pass a path to a `*.json` class map
#       as the **last** argument. A class map defines how labels/IDs map to class
#       names (and optionally groups). Example keys: `{ "1": "Forest", "2": "Water" }`.
#     
#         Examples:
#         - `accuracy cross.csv class_map.json`
#         - `accuracy raster.tif reference_raster.tif class_map.json`
#         - `accuracy raster.tif reference_vector.gpkg class_map.json`
#     
#     2.5. Special case - single raster only:
#         - classified image - usually `*.tif`
#         - `Warning!!!` If a reference file with the same base name exists in the
#           same folder with the suffix `_ref`, it will be used automatically.
#     
#         Examples:
#         - `accuracy raster.tif` - if `raster_ref.tif` exists
#         - `accuracy raster.tif` - if `raster_ref.shp` exists
#         - `accuracy raster.tif` - if `raster_ref.gpkg` exists
# """
# 
# 
# 
# info_metrics = """
# 1. Notacja binarna
# The definitions of the metrics are mainly based on the binary \
# error matrix with the following symbols:
#   - `TP` true positive
#   - `TN` true negative
#   - `FP` false positive
#   - `FN` false negative.
# 
# 2. Remote sensing 
# Accuracy metrics classically used in remote sensing:
#   - OA (overall_accuracy):
#    -- OA = sum(TP) / (TP + TN + FP + FN)
# 
#   -  PA (producer_accuracy):
#    -- PA = TP / (TP + FN)
# 
#   -  UA (user_accuracy)
#      -- UA = TP / (TP + FP)
# 
#   -  OME (omission errors / errors_of_omission):
#      -- OME = FN / (TP + FN)
# 
#   -  CME (errors_of_commision):
#      -- CME = FP / (TP + FP)
# 
#   -  NPV (negative predictive value):
#      -- NPV = TN/(TN + FN) = 1 − FOR
# 
# 3. Contemporary classification accuracy metrics 
# Classification accuracy metrics found in contemporary scientific \
# publications (some metrics overlap with some of the metrics mentioned in \
# `point 2`).
# 
# These metrics can be conventionally divided into `simple` metrics \
# (calculated directly from the TP, TN, FP and FN values) and `complex` metrics \
# (calculated using simple metrics).
# 
# 3.1. Simple metrics:
# 
#    -   ACC (accuracy):
#       -- ACC = (TP+TN)/(P+N) = (TP+TN)/(TP+TN+FP+FN)
# 
#    -  PPV (precision or positive predictive value):
#       -- PPV = TP / (TP + FP)
# 
#    -  PPV (precision or positive predictive):
#       -- PPV = TP / (TP + FP)
# 
#    -  TPR (sensitivity, recall, hit rate, or true positive rate):
#       -- TPR = TP/P = TP/(TP + FN) = 1 − FNR
# 
#    -  TNR (specificity, selectivity or true negative rate):
#       -- TNR = TN/N = TN/(TN + FP) = 1 − FPR
# 
#    -  NPV (negative predictive value):
#       -- NPV = TN/(TN + FN) = 1 − FOR
# 
#    -  FNR (miss rate or false negative rate):
#       -- FNR = FN/P = FN/(FN + TP) = 1 − TPR
# 
#    -  FPR (fall-out or false positive rate):
#       -- FPR = FP/N = FP/(FP + TN) = 1 − TNR
# 
#    -  FDR (false discovery rate):
#       -- FDR = FP/(FP + TP) = 1 − PPV
# 
#    -  FOR (false omission rate):
#       -- FOR = FN/(FN + TN) = 1 − NPV
# 
#    -  TS / CSI (Threat score (TS) or critical success index (CSI)):
#       -- TS = TP/(TP + FN + FP)
# 
#    -  MCC (Matthews correlation coefficient):
#       -- mcc = (TP*TN - FP*FN) / [(TP+FP) * (TP+FN) * (TN+FP) * (TN+FN)]^0.5
# 
# 3.2. Complex metrics:
#    - PT (Prevalence Threshold):
#      -- PT = [(TPR*(1 − TNR))^0.5 + TNR − 1] / (TPR + TNR − 1)
# 
#    - BA (Balanced accuracy):
#      -- ba = (TPR + TNR)/2
# 
#    - F1 score (is the harmonic mean of precision and sensitivity):
#      -- f1 = 2*(PPV*TPR)/(PPV+TPR) = (2*TP)/(2*TP+FP+FN)
# 
#    - FM (Fowlkes–Mallows index):
#      -- fm = [(TP/(TP+FP))*(TP/(TP+FN))]^0.5 = (PPV * TPR)^0.5
# 
#    - BM (Bookmaker informedness):
#      -- bm = TPR + TNR - 1
# 
#    - MK (markedness (MK) or deltaP):
#      -- mk = PPV + NPV - 1
# """
# 
# 
# info_data = """
# 1. Raw data:
# Stores classification results in a table (*.csv) with 2 or 3 columns:
# 
#    - 2 columns:
# 
#              |    true   | predicted |
#              |-----------+-----------|
#              |    int    |    int    |
#              |    ...    |    ...    |
# 
#    - 3 columns:
# 
#              |    true   | predicted |  lables  |
#              |-----------+-----------+----------|
#              |    int    |    int    |    str   |
#              |    ...    |    ...    |    ...   |
# 
#    Where:
#        - first column: true values (actual classes)
#        - second column: predicted values (predicted classes)
#        - third column: short names of classes e.g. water, wheat etc.
# 
#     Input:
#         - columns must be in order [true_values, predicted]
#         - column names do not matter (eg. true, true_values, etc)
# 
# 
# 2. Raw - cross matrix:
# Confusion matrix for multi-class classification:
#   - contains only numbers: no column or row descriptions, no summaries
#   - is square: classes in columns must correspond to classes in rows, even if \
#   there are zeros in some class
# 
#   Default layout is:
#     - rows: True classes (true labels).
#     - columns: Predicted classes (predicted labels)
# 
#        |   21  |    5   |   7   | ...
#        |    6  |   31   |   2   | ...
#        |    0  |    1   |  22   | ...
#        |  ...  |   ...  |  ...  | ...
# 
# 
# 3. Cross - cross matrix:
# Confusion matrix for multi-class classification:
#   - contains numbers and descriptions of columns and rows (class names), \
#     without summaries
#   - does not have to be square:
# 
#   Default layout is:
#     - rows: True classes (true labels).
#     - columns: Predicted classes (predicted labels)
# 
#        |            | water | forest | urban | ...
#        |------------+-------+--------+-------+-----
#        |   water    |   21  |    5   |   7   | ...
#        |   forest   |    6  |   31   |   2   | ...
#        |   urban    |    0  |    1   |  22   | ...
#        |    ...     |  ...  |   ...  |  ...  | ...
# 
#   
# 4. Full - cross matrix:
# Full confusion matrix for multi-class classification:
#   - contains numbers, column and row descriptions (class names) and row and \
#     column summaries
#   - does not have to be square:
# 
#   Default layout is:
#     - rows: True classes (true labels).
#     - columns: Predicted classes (predicted labels)
# 
#        |            | water | forest | urban | ... |  sums  |
#        |------------+-------+--------+-------+-----|--------|
#        |   water    |   21  |    5   |   7   | ... |   ...  |
#        |   forest   |    6  |   31   |   2   | ... |   ...  |
#        |   urban    |    0  |    1   |  22   | ... |   ...  |
#        |    ...     |  ...  |   ...  |  ...  | ... |   ...  | 
#        |------------+-------+--------+-------+-----|--------|
#        |    sums    |  ...  |   ...  |  ...  | ... |   ...  |
# 
# 
# 5. Binary - cross matrix:
# Confusion matrix for multi-class classification.
# 
#                       |    | water | forest | ... |
#                       |----+-------+--------+-----|
#                       | TP |    1  |   55   | ... |
#                       | TN |   15  |   99   | ... |
#                       | FP |    5  |    3   | ... |
#                       | FN |   33  |   46   | ... |
# 
#     where:
#   - columns: represent the classes in the dataset
#   - rows: represent different types of classification outcomes for each class:
#   - TP (True Positives): the number of samples correctly classified as a given class
#   - TN (True Negatives): the number of samples that do not belong to a given class and were correctly identified as not belonging.
#   - FP (False Positives): the number of samples incorrectly classified as a given class
#   - FN (False Negatives): the number of samples of a given class that were incorrectly classified as not belonging to that class
# 
# 
# 6. Raster data
# The input data can also be raster images and vector data. You can use:
#  - two raster images: classification result and reference image (mask)
#  - raster image and vector data
# 
# Raster images should be in `*.tif` format, georeferenced. Different file extensions are accepted:
#  - '*.tif', '*.tiff', '*.TIF', '*.TIFF'
# 
# Vector data: two popular formats are accepted:
#  - `*.shp` EERI shapefile spatial data format
#  - `*.gpkg` the GeoPackage (GPKG)
# 
# 
# 6.1 Tip:
# If the reference data file has the same name as the classification result file with an additional
# suffix `_ref`, then you just need to provide the image file address (classification result) as input
# and the script will search for the reference data, e.g.:
# 
# Instead of typing:
#  - accuracy my_classification.tif my_classification_ref.tif
#  - accuracy my_classification.tif my_classification_ref.shp
# 
# You can:
#  - accuracy my_classification.tif
#  - accuracy my_classification.tif
# 
# 
# 7. `*.json` file
# A JSON text file used to rename (relabel) classes. The file must map current
# class labels to new labels, for example:
# 
# {
#   "cl_a": "grass",
#   "cl_b": "coniferous forest",
#   "cl_c": "roads"
# }
# 
# If the input data do not include class names (for example, `cross_raw.csv`),
# the mapped names will be applied based on the row and column indices (integers)
# of the matrix.
# 
# 
# """
# 
# info_formula = """
# 1. Calculation formula
# You can define your own calculation formula:
# 
#  - The calculations use the binary_cross matrix table.
#  - The formula must follow Python's arithmetic syntax.
#  - Use the following labels: TP, TN, FP, and FN.
#  - The formula should consist of a left-hand side and a right-hand side: `metric = mathematical operations`. Example:
#    -- mcc=(TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
#  - Do not use whitespace (e.g., spaces) in the formula or metric name.
#  - The metric name should be a short string, such as OA, f1, etc.
#  - The pattern entered into the script must be surrounded by quotation marks (single `'` or double `"`).
# 
# 2. Example of script use:
# 
#  - accuracy input_path -f "mcc=(TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5"
#  - accuracy input_path --formula "mcc=(TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5"
# 
# """


# funkcje do przetwarzania tekstu pomocy dla streamlit (uproszczony
# markdown)

def mk_tables(txt: list[str]) -> list[str]:
    """Przetwarza wiersze zawierające tabele, czyli zaczynające się od `|`."""
    res = []
    txt = txt[:]

    # start: False - wiersz to nie tabela, True - to tabela
    start = False

    for line in txt:
        if line.startswith('|'):
            # początek tabeli
            if not start:
                start = True
                res.append("  ```bash")
                
            line = f"    {line}"
            res.append(line)
        else:
            # breakpoint()
            if start:
                start = False
                res.append("  ```")
            res.append(line)
    
    return res


def mk_equations(txt: list[str]) -> list[str]:
    """Przetwarza linie zawierające równania, czyli wiersze zaczynające
    się od `--`.
    """
    res = []
    txt = txt[:]

    for line in txt:
        if line.startswith('--'):
            line = line.replace('--', '')
            line = line.strip()
            line = '\n' + f"$$\n{line}\n$$"
            line = line.replace(r'^0.5', r'^{0.5}')
        res.append(line)
    return res


def get_index(txt: list[str], pattern: str) -> list[str]:
    """Przechodzi linia po linii i wyszukują tę linię która pasuje do
    wzorca. Zwraca index tej linii.
    """
    idx = None
    txt = txt[:]

    for i, line in enumerate(txt):
        if re.search(pattern, line):
            idx = i
    return idx


class Readme2HelpCli:
    """Klasa służy do konwersji pliku README.md na tekst pomocy wyświetlany
    przez skrypt w konsoli tekstowej.
    """
    chapters = {
            'usage_help': '# Usage help',
            'data_help': '# Data help',
            'metrics_help': '# Metrics help',
            'formula_help': '# Formula help'}

    patterns = OrderedDict({
        'h1': re.compile(r'^#\s+'),
        'h2': re.compile(r'^##\s+'),
        'h3': re.compile(r'^###\s+'),
        'empty': re.compile(r"^\s*$"),
        'hash': re.compile(r"^\s+#\s.+"),
        'bash': re.compile(r'.*```.*$'),
        'bold': re.compile(r'^\s{1,3}[*]{2}.*[*]{2}'),
        'lists': re.compile(r'^\s*-\s+|^\s*--\s|^\s*+\s'),
        'colon': re.compile(r':$'),
        'quote': re.compile(r'^\s*>\s*'),
        'table': re.compile(r'^\s*\|'),
        'img': re.compile(r"^\s*!\[(.+)\]"),
        'equation': re.compile(r"^\s*[$]{2}"),
        # 'usage_help': re.compile(r"^\s{3,}.+#.+"),
        'terminal': re.compile(r"^\s*-:[$]\s.+"),
        'json': re.compile(r'\s*{\s*$|\s*}\s*$'),
        })

    def __init__(self, txt, indent=4, width=110):
        """Args:
          - txt:  str, from README.md
          - indent:  int, ile spacji wcięcia całego tekstu
          - width: int, całkowita długość linii (szerokość tekstu)
        """
        # ustawia atrybuty dla instancji: klucze z chapters
        for key in type(self).chapters:
            setattr(self, key, None)
        self.patterns = type(self).patterns.copy()
        self.indent = indent
        self.width = width
        self.global_indent = 4

        self.txt = [line.rstrip() for line in txt.splitlines()]
        self._split_chapters(self.txt)

        for chapter_name in type(self).chapters:
            # numeracja nagłówków przetwarzanego README
            self.h2_idx = 0
            self.h3_idx = 1

            chapter_txt = getattr(self, chapter_name).splitlines()
            # groups = self._split_groups(getattr(self, chapter_name).splitlines())
            groups = self._split_groups(chapter_txt)
            help_txt = self._format_groups(groups)
            setattr(self, chapter_name, '\n'.join(help_txt))
        
        del self.h2_idx, self.h3_idx

    def _split_chapters(self, lines: list[str]) -> list[dict]:
        """Dzieli tekst z README.md na osobne rozdziały, zgodnie z atrybutem
        `chapters`.
        """
        # chapters: {'key': 'pattern'}
        chapters = type(self).chapters.copy()
        res = {key: [] for key in chapters.keys()}

        # reverse chapters {'pattern': 'key'}
        chapters = dict(zip(chapters.values(), chapters.keys()))
        curr_chapt = None
        
        for line in lines:
            # pomija linie '---'
            if re.search(r'---', line):
                continue

            # usuwa cytowania `>`
            line = line.replace('>', '', 1)

            if re.search(r"^#\s.+", line) and chapters.get(line, False):
                curr_chapt = chapters.get(line)
            elif re.search(r"^#\s.+", line):
                curr_chapt = None

            # jeśli aktualnie jest wykrywany jakiś chapter
            if curr_chapt is not None:
                # sprawdź czy to linia nie zaczyna nowego chaptera
                check = chapters.get(line, None)
                if check is None:
                    res[curr_chapt].append(line)
                    continue
                else:
                    curr_chapt = check
                    continue
        
        for chapter_name, lines in res.items():
            lines = '\n'.join(lines).strip()
            setattr(self, chapter_name, lines)
            
    def _split_groups(self, lines: list[str]) -> list[dict]:
        """Dzieli tekst na grupy (nagłówki, listy, ...) do których stosuje
        odpowiednie formatowanie tekstu.
        """
        res = []
        current_group = False 
        patterns = self.patterns.copy()
        pattern_json = patterns.pop('json')
        reverse_patterns = dict(zip(patterns.values(),
                                    patterns.keys())
                                )
    
        for line in lines:
            # 1. usuwan znaki `>` cytowania
            if re.search(r"^\s*[>]", line):
                line = line.replace(">", '', 1).strip()
            
            # 2. szuka json
            if pattern_json.search(line):
                # breakpoint()
                if current_group != 'json':
                    # rozpoczyna block json `{`
                    res.append({'json': line})
                    current_group = 'json'
                elif current_group == 'json':
                    # kończy blok json `}`
                    res[-1]['json'] += '\n' + line
                    current_group = False
                continue

            # jeśli json to znaczy że blok json jest otwarty i ma być
            # kontynuowany
            if current_group == 'json':
                res[-1]['json'] += '\n' + line
                continue

            # 3. Sprawdza dopasowania pojedynczych linii
            for pattern, name in reverse_patterns.items():
                if pattern.search(line):
                    if name == 'colon' and current_group == 'paragraf':
                        continue

                    current_group = False
                    if name == 'bash':
                        break

                    res.append({name: line})
                    break
            
            # dziwna wersja pętli `for`!!!
            # 4. Jeśli do tego momentu nie ma dopasowania -> paragraf
            else:
                if current_group == 'paragraf':
                    res[-1]['paragraf'] += ' ' + line.strip()
                elif current_group != 'paragraf':
                    current_group = 'paragraf'
                    res.append({'paragraf': line.strip()})

        return res

    def _format_groups(self, groups: list[dict]):  #, n, width):
        res = []
        # pamięta poprzednią grupę
        name_mem = None
        n = self.global_indent 

        for gr in groups:
            name = list(gr.keys())[0]
            method = getattr(self, f"_format_{name}")
            txt = method(gr[name], name=name_mem)
            if name != 'json':
                txt = f"{n * ' '}{txt}"

            try:
                res.append(txt)
            except:
                print(f"\n\nError:\n{gr}\n\n")
                import sys
                sys.exit(1)
            name_mem = name
        res += '\n'
        return res

    def _format_h1(self, txt, **kwargs):
        # txt: '# Some title'
        txt = txt[1:].strip()
        return txt

    def _format_h2(self, txt, **kwargs):
        # txt: '## Some title'
        self.h2_idx += 1
        self.h3_idx = 1
        txt = txt[2:].strip()
        txt = f"{self.h2_idx}. {txt}"
        return txt

    def _format_h3(self, txt, **kwargs):
        # txt: '### Some title'
        txt = txt[3:].strip()
        txt = f"  {self.h2_idx}.{self.h3_idx}. {txt}"
        self.h3_idx += 1
        return txt

    def _format_img(self, line, **kwargs):
        key = self.patterns['img'].search(line).group(1)
        # `imgs_replacement`: zmienna globalna modułu
        # txt = f"{3*' '}{imgs_replacement[key]}"
        line = imgs_replacement[key]
        line = self._format_as_list(line)
        return line

    def _format_equation(self, line: str, **kwargs):
        name = kwargs.get('name')
        n = 4
        if name == 'lists':
            n=6
        line = f"{n*' '}{line.strip()}"
        return line

    # def _format_numerowany(self, txt, **kwargs):  #, n, width):
    #     """Args:
    #     - txt:  '1. line\n2. line\n3.line..'
    #     - n:  wielkość wcięcia listy: n * ' '
    #     - w:  długość maksymalna linii
    #     """
    #     # ['1. line', '2. line', ...]
    #     txt = [line.strip() for line in txt.splitlines()]

    #     # zamienia kilkukrotne spacje na pojedyncze np. 'abc    xx` -> 'abc xx'
    #     txt = [" ".join(line.split()) for line in txt]

    #     # zawija długie linie
    #     txt = [
    #         textwrap.fill(line,
    #                       width=self.width,
    #                       subsequent_indent=3 * " ") for line in txt
    #     ]

    #     txt = "\n".join(txt)
    #     return txt

    def _format_bold(self, line: str, **kwargs):
        return line

    def _format_colon(self, line: str, **kwargs):
        return f" {line}"

    def _format_empty(self, txt, **kwargs):
        return ''

    def _format_as_list(self, line: str, **kwargs):
        """Formatuje dowolną linię jako listę: np.:
          - line = 'abc 345' -> '    - abc 345'
        """
        line = line.strip()
        width = self.width
        n = self.indent

        if line.startswith('-'):
            line = f"{n * ' '}{line}"
        else:
            line = f"{n * ' '}- {line}"

        line = textwrap.fill(line,
                             width=width,
                             subsequent_indent=(3 + n) * " "
                             )
        return line

    def _format_lists(self, line, **kwargs):
        return self._format_as_list(line)

    def _format_usage_help(self, txt, **kwargs):
        return self._format_lists(txt)

    def _format_terminal(self, line: str, **kwargs):
        return self._format_as_list(line)

    def _format_hash(self, line, **kwargs):
        line = line.replace('#', '')
        return self._format_as_list(line)

    def _format_paragraf(self, txt, **kwargs):  #, width):
        """Zwykły tekst wieloliniowy, składany jako akapit tekstu."""
        txt = [line.strip() for line in txt.splitlines()]
        txt = [" ".join(line.split()) for line in txt]
        txt = " ".join(txt)
        txt = textwrap.fill(txt,
                            width=self.width,
                            initial_indent=' ',
                            subsequent_indent=4 * " ")
        return txt

    def _format_table(self, table, **kwargs):  #, n):
        n = self.indent + 4
        table = [line.strip() for line in table.splitlines()]
        table = [f'{" "*n}{line}' for line in table]
        table = "\n".join(table)
        return table

    def _format_json(self, txt, **kwargs):
        txt = json.loads(txt)
        txt = json.dumps(txt, ensure_ascii=False, indent=2)
        n = self.global_indent + 4
        txt = textwrap.indent(txt, n*' ')
        return txt


class Readme2Streamlit(Readme2HelpCli):
    """Klasa dzieli tekst odczytany z pliku README.md na potrzeby wyświetlania
    w app.py.
    """
    ...

    # def __init__(self, txt, indent=4, width=110):
    def __init__(self, txt, docs_path):
        """Args:
            - txt:  str, text from README.md
            - docs_path:  str, path to folder with images (screeny z konsoli
                          pokazujące przykłady użycia)
        """
        # ustawia atrybuty dla instancji: klucze z chapters
        for key in type(self).chapters:
            setattr(self, key, None)
        self.patterns = type(self).patterns.copy()

        self.txt = [line.rstrip() for line in txt.splitlines()]
        self._split_chapters(self.txt)
        self.docs_path = Path(docs_path)

        # dostosowuje metrics_help do streamlit
        self.metrics_help = self._format_metrics_help()

        # rozwiąż ścieżki do obrazków
        self.usage_help = self._resolve_img_paths()

    def _format_metrics_help(self):
        res = []
        txt = self.metrics_help.splitlines()
        pattern = self.patterns.get('equation')
        for line in txt:
            if pattern.search(line):
                line = '\n' + line
            res.append(line)
        res = '\n'.join(res)
        return res

    def _resolve_img_paths(self):
        res = []
        txt = self.usage_help.splitlines()
        # pattern = re.compile(r"^\s*!\[.+]\((.+)\)")
        pattern = re.compile(r"(^\s*!\[.+]\()(.+)(\))")

        for line in txt:
            if pattern.search(line):
                img_pth = pattern.search(line).group(2)
                name = Path(img_pth).name
                img_pth = (self.docs_path / name).resolve()
                if img_pth.exists():
                    data = base64.b64encode(img_pth.read_bytes()).decode()
                    ext = img_pth.suffix.lstrip(".")
                    line = (f"<img alt='{name}' "
                            f"src='data:image/{ext};base64,{data}' "
                            " style='max-width:50%; border-radius:8px;'>")
                # line = pattern.sub(fr'\1{pth}\3', line)
                # print(line)
        
            res.append(line)
        res = '\n'.join(res)
        return res
        




def parse_help_text(txt):
    """Funkcja przetwarza tekst pomocy na markdown używany w streamlit."""
    txt = txt.splitlines()
    txt = [line.strip() for line in txt]
    if txt[0] == '':
        txt = txt[1:]

    txt = [f"#### {line}" if re.search(r'^\d\.\d', line) else
           line for line in txt]
    txt = [f"### {line}" if re.search(r'^\d', line) else line for line in txt]
    txt = ['\n' if line == '' else line for line in txt]

    # txt = [f"\t{line}" if line.startswith('|') else line for line in txt]
    txt = mk_tables(txt)
    
    # przetwarza równania (linie zaczynają si e od --) 
    txt = mk_equations(txt)
    txt = [f"  {line}" if line.startswith('-') else line for line in txt]


    # format jednej konkretnej linii
    # pat = "- '*.tif', '*.tiff', '*.TIF', '*.TIFF'"
    pat = r"\*\.TIFF"
    idx = get_index(txt, pat)
    if idx is not None:
        txt[idx] =  "\n  ```bash \n  #'*.tif', '*.tiff', '*.TIF', '*.TIFF'\n```"
    
    txt = '\n'.join(txt)
    return txt
