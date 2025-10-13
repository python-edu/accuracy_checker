# About the script

This script computes popular **classification accuracy metrics**. It was written to facilitate the comparison of results across different scientific studies. Published raster-image classification results in the Earth sciences often come from diverse methodologies and tools, which makes direct comparison difficult.

If you have any of the following data types:
- a post-classification image,
- raw classification pairs (`true`, `predicted`),
- a confusion matrix (cross matrix) or a per-class binary confusion matrix,

you can use this tool to compute a range of accuracy metrics.

---

# Usage

After installation, run the script from any terminal by typing the command and providing the required positional
arguments. In the examples below, the argument is the name of a `csv` file with classification results
(see **Data Help**).

1. ```bash
     @: accuracy raw_classification.csv
   ```
   *If installed via the installer, you can run it from anywhere without manually activating the virtual environment.*

2. ```bash
     (acc) @: accuracy raw_classification.csv
   ```
   *If you installed manually into a virtual environment, activate the environment first, then run the command.*

There is also a simple GUI based on **Streamlit** that opens in your web browser:

3. ```bash
     @: accuracy_gui
   ```
   **Note**:
    - *the GUI may not expose the full functionality of the CLI.*

---

# Installation

You can install the script manually or use the installer.

First, download or clone the repository to your local drive:
  ```bash
    git clone https://github.com/python-edu/accuracy_checker.git
  ```

Unpack the repository to your target location. Open a terminal and go to the root directory of the unpacked repo (acc):
  ```bash
    cd acc/
  ```


## Installer

Run the installer by entering the command:
```bash
  python install.py
```

The installer will:

  - create a new virtual environment,
  - install dependencies from `requirements.txt`,
  - install the package into that environment,
  - create `$HOME/.local/bin` (Linux/macOS) or `%USERPROFILE%\bin` (Windows) if it doesn’t already exist,
  - create launcher scripts in that `bin` directory so you can run the program from anywhere,
  - add that `bin` directory to the user’s `PATH`.

**Note**:
  - *The installer usually refreshes your PATH automatically. Sometimes you’ll need to close and reopen your terminal
  for the changes to take effect.*



## Manual installation

   ```bash
     # 1. Create a virtual environment, e.g.:
     python -m venv env --prompt acc
     
     # 2. Activate the virtual environment and install dependencies, e.g.:
     env/Scripts/activate       # Windows System:
     source env/bin/activate    # Linux (Debian):
     python -m pip install -r requirements.txt
     
     # 3. Install the package in the environment:
     python -m pip install -e .
   ```

## Uninstallation
To completely uninstall the script, including removing the unpacked repository, run the installer with the purge
option (`-p/--purge`):
  ```bash
    python install.py -p
  ```

---


# Usage help
You can display help by calling the script with the following options:
  ```bash 
    accuracy -h/--help      # general help on how the script works
    
    accuracy usage help     # help about script usage

    accuracy data help      # information about input data

    accuracy metrics help   # information about accuracy metrics

    accuracy formula help   # help for custom calculation formulas
  ```


## Single file
The input is a single `*.csv` file:
 - raw data: `data2cols.csv` or `data3cols.csv`
 - confusion matrix: cross_raw.csv, cross.csv or cross_full.csv
 - binary_cross.csv

```bash
  accuracy file.csv
  accuracy file.csv class_map.json
```


## Raster / vector
The input is a raster or / and vector:
 - an image, after classification - typically `*.tif`
 - reference data (reference raster mask): image/mask `*.tif` or vector data e.g. `*.shp`, `*.gpkg`

 ```bash
   accuracy raster.tif
   accuracy raster.tif class_map.json
   accuracy raster.tif reference_raster.tif
   accuracy raster.tif reference_raster.tif class_map.json
   accuracy raster.tif reference_vector.shp
   accuracy raster.tif reference_vector.shp class_map.json
 ```

## Custom formula
You can enter your own formula to estimate accuracy:
 ```bash
   accuracy file.csv -f "ac = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN))**0.5"
 ```

---


# Data Help

## Raw data:
Stores classification results in a `*.csv` table with 2 or 3 columns:

```bash
    # 2 columns:

    |    true   | predicted |
    |-----------+-----------|
    |    int    |    int    |
    |    ...    |    ...    |


    # 3 columns:

    |    true   | predicted |  lables  |
    |-----------+-----------+----------|
    |    int    |    int    |    str   |
    |    ...    |    ...    |    ...   |
```

 **Where**:
 - first column: true values (actual classes)
 - second column: predicted values (predicted classes)
 - third column: short names of classes e.g. water, wheat etc.

 **Input requirements**:
 - columns must be in order `[true_values, predicted]`
 - column names do not matter (eg. true, true_values, etc)


## Raw cross matrix
Confusion matrix for multi-class classification:
 - contains `only numbers`: no column / row descriptions (labels), no summaries (totals)
 - must be square: classes in columns must correspond to classes in rows, even if there are zeros in some class.

Default layout is:
 - rows: True classes (true labels).
 - columns: Predicted classes (predicted labels).

```bash
    |   21  |    5   |   7   | ...
    |    6  |   31   |   2   | ...
    |    0  |    1   |  22   | ...
    |  ...  |   ...  |  ...  | ...
```


## Cross (labeled cross matrix)
Confusion matrix for multi-class classification:
 - contains numbers and descriptions (labels) of columns and rows (class names), without summaries (totals)
 - does not have to be square.

Default layout is:
 - rows: True classes (true labels).
 - columns: Predicted classes (predicted labels).

```bash
    |            | water | forest | urban | ...
    |------------+-------+--------+-------+-----
    |   water    |   21  |    5   |   7   | ...
    |   forest   |    6  |   31   |   2   | ...
    |   urban    |    0  |    1   |  22   | ...
    |    ...     |  ...  |   ...  |  ...  | ...
```


## Full cross matrix
Full confusion matrix for multi-class classification:
 - contains numbers, column and row descriptions (class names) and row and column summaries
 - does not have to be square.

Default layout is:
 - rows: True classes (true labels).
 - columns: Predicted classes (predicted labels)

```bash
    |            | water | forest | urban | ... |  sums  |
    |------------+-------+--------+-------+-----|--------|
    |   water    |   21  |    5   |   7   | ... |   ...  |
    |   forest   |    6  |   31   |   2   | ... |   ...  |
    |   urban    |    0  |    1   |  22   | ... |   ...  |
    |    ...     |  ...  |   ...  |  ...  | ... |   ...  |
    |------------+-------+--------+-------+-----|--------|
    |    sums    |  ...  |   ...  |  ...  | ... |   ...  |
```


## Binary cross matrix
Per-class binary confusion values.

```bash
    |    | water | forest | ... |
    |----+-------+--------+-----|
    | TP |    1  |   55   | ... |
    | TN |   15  |   99   | ... |
    | FP |    5  |    3   | ... |
    | FN |   33  |   46   | ... |
```

 **where**:
 - columns: classes present in the dataset
 - rows: binary outcomes per class
 - TP (True Positives): the number of samples correctly classified as a given class
 - TN (True Negatives): the number of samples that do not belong to a given class and were correctly identified as not belonging.
 - FP (False Positives): the number of samples incorrectly classified as a given class
 - FN (False Negatives): the number of samples of a given class that were incorrectly classified as not belonging to that class


## Raster data
The input data can also be raster images and vector data:
 - two raster images: classification result and reference image (mask)
 - raster image and vector data


1. Raster images:
   - should be in `*.tif` format, georeferenced
   - different file extensions are accepted: `'*.tif', '*.tiff', '*.TIF', '*.TIFF'`

2. Vector data formats supported:
 - `*.shp` ESRI Shapefile spatial data format
 - `*.gpkg` the GeoPackage (GPKG)

> **Tip**:
>
> If the reference file shares the same name as the classification result raster with an `_ref` suffix, you can provide
only the classification raster path and the script will find the reference automatically. For example, instead of:
> ```bash
>   accuracy my_classification.tif my_classification_ref.tif
>   accuracy my_classification.tif my_classification_ref.shp
> ```
> 
> You can:
> ```bash
>   accuracy my_classification.tif
> ```

---


# Metrics help

## Symbols

We use the standard binary symbols:
 - TP true positive
 - TN true negative
 - FP false positive
 - FN false negative.


## Remote sensing metrics
Accuracy metrics classically used in remote sensing:

 - OA (overall_accuracy):
   $$OA = \sum(TP) / (TP + TN + FP + FN)$$

 - PA (producer_accuracy):
   $$PA = TP / (TP + FN)$$

 - UA (user_accuracy)
   $$UA = TP / (TP + FP)$$

 - OME (omission errors / errors_of_omission):
   $$OME = FN / (TP + FN)$$

 - CME (errors_of_commision):
   $$CME = FP / (TP + FP)$$

 - NPV (negative predictive value):
   $$NPV = TN/(TN + FN) = 1 - FOR$$


## Modern accuracy
> Classification accuracy metrics found in contemporary scientific publications (*some overlap with those above*).

These metrics can be conventionally divided into simple metrics (calculated directly from the TP, TN, FP and FN values)
and complex metrics (calculated using simple metrics).

### 1. Simple metrics

  - ACC (accuracy):
    $$ACC = (TP+TN) / (P+N) = (TP+TN) / (TP+TN+FP+FN)$$
 
  - PPV (precision or positive predictive value):
    $$PPV = TP / (TP + FP)$$
 
  - TPR (sensitivity, recall, hit rate, or true positive rate):
    $$TPR = TP / P = TP / (TP + FN) = 1 - FNR$$
 
  - TNR (specificity, selectivity or true negative rate):
    $$TNR = TN / N = TN / (TN + FP) = 1 - FPR$$
 
  - NPV (negative predictive value):
    $$NPV = TN / (TN + FN) = 1 - FOR$$
 
  - FNR (miss rate or false negative rate):
    $$FNR = FN / P = FN / (FN + TP) = 1 - TPR$$
 
  - FPR (fall-out or false positive rate):
    $$FPR = FP / N = FP / (FP + TN) = 1 - TNR$$
 
  - FDR (false discovery rate):
    $$FDR = FP / (FP + TP) = 1 - PPV $$
 
  - FOR (false omission rate):
    $$FOR = FN / (FN + TN) = 1 - NPV $$
 
  - TS / CSI (Threat score (TS) or critical success index (CSI)):
    $$TS = TP / (TP + FN + FP) $$
 
  - MCC (Matthews correlation coefficient):
    $$mcc = (TP \cdot TN - FP \cdot FN) / ((TP+FP) \cdot (TP+FN) \cdot (TN+FP) \cdot (TN+FN))^{0.5}$$


### 2. Complex metrics

 - PT (Prevalence Threshold):
   $$pt = ((TPR \cdot (1 - TNR))^{0.5} + TNR - 1) / (TPR + TNR - 1)$$

 - BA (Balanced accuracy):
   $$ba = (TPR + TNR) / 2$$

 - F1 score (is the harmonic mean of precision and sensitivity):
   $$f1 = 2 \cdot (PPV \cdot TPR) / (PPV + TPR) =  2\cdot TP / (2 \cdot TP + FP + FN)$$

 - FM (Fowlkes–Mallows index):
   $$fm = ((TP/(TP+FP)) \cdot (TP/(TP+FN)))^{0.5} = (PPV \cdot TPR)^{0.5}$$

 - BM (Bookmaker informedness):
   $$bm = TPR + TNR - 1$$

 - MK (markedness or deltaP):
   $$mk = PPV + NPV - 1$$


# Dependencies

 - [pytexit](https://pytexit.readthedocs.io/), tabulate, jinja2
 - numpy, pandas, geopandas
 - shapely, fiona, pyproj
 - rasterio, rtree
 - streamlit, streamlit-navigation-bar



# LICENSE
The project is licensed under the [MIT](./LICENSE) license.
For details, see the `LICENSE` file.
