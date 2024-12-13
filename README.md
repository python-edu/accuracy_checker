# About script

The script's task is to calculate popular classification accuracy metrics. The script was written in connection with the
need to compare the results of various scientific studies. Published raster image classification results in the fields
of earth science come from various research methodologies and various tools, which often prevents easy comparison of
results.

If data of the type: image after classification, raw classification results (true, predicted), error matrix or binary
error matrix is available, the script can be used to calculate various accuracy metrics.


# Usage help

Running the script, order and layout of input files:

### 1. The input is a single `*.csv` file:
 - raw data: data2cols.csv or data3cols.csv
 - confusion matrix: cross_raw.csv, cross.csv or cross_full.csv
 - binary_cross.csv

 >- `accuracy file.csv`
 >- `accuracy file.csv class_map.json`


### 2. Input raster / vector:
 - an image, after classification - usually of type `*.tif`
 - reference data: image/mask `*.tif` or vector data e.g. `*.shp`, `*.gpkg`

 >- `accuracy raster.tif`
 >- `accuracy raster.tif class_map.json`
 >- `accuracy raster.tif reference_raster.tif`
 >- `accuracy raster.tif reference_raster.tif class_map.json`
 >- `accuracy raster.tif reference_vector.shp`
 >- `accuracy raster.tif reference_vector.shp class_map.json`


# Data Help

### 1. Raw data:
Stores classification results in a table (`*.csv`) with 2 or 3 columns:

```
    - 2 columns:

    |    true   | predicted |
    |-----------+-----------|
    |    int    |    int    |
    |    ...    |    ...    |

    - 3 columns:

    |    true   | predicted |  lables  |
    |-----------+-----------+----------|
    |    int    |    int    |    str   |
    |    ...    |    ...    |    ...   |
```

 *Where*:
 - first column: true values (actual classes)
 - second column: predicted values (predicted classes)
 - third column: short names of classes e.g. water, wheat etc.

 *Input*:
 - columns must be in order `[true_values, predicted]`
 - column names do not matter (eg. true, true_values, etc)


### 2. Raw - cross matrix:
Confusion matrix for multi-class classification:
 - contains only numbers: no column or row descriptions, no summaries
 - is square: classes in columns must correspond to classes in rows, even if there are zeros in some class

Default layout is:
 - rows: True classes (true labels).
 - columns: Predicted classes (predicted labels)

``` |   21  |    5   |   7   | ...
    |    6  |   31   |   2   | ...
    |    0  |    1   |  22   | ...
    |  ...  |   ...  |  ...  | ...
```


### 3. Cross - cross matrix:
Confusion matrix for multi-class classification:
 - contains numbers and descriptions of columns and rows (class names), without summaries
 - does not have to be square:

Default layout is:
 - rows: True classes (true labels).
 - columns: Predicted classes (predicted labels)

```
    |            | water | forest | urban | ...
    |------------+-------+--------+-------+-----
    |   water    |   21  |    5   |   7   | ...
    |   forest   |    6  |   31   |   2   | ...
    |   urban    |    0  |    1   |  22   | ...
    |    ...     |  ...  |   ...  |  ...  | ...
```


### 4. Full - cross matrix:
Full confusion matrix for multi-class classification:
 - contains numbers, column and row descriptions (class names) and row and column summaries
 - does not have to be square:

Default layout is:
 - rows: True classes (true labels).
 - columns: Predicted classes (predicted labels)

```
    |            | water | forest | urban | ... |  sums  |
    |------------+-------+--------+-------+-----|--------|
    |   water    |   21  |    5   |   7   | ... |   ...  |
    |   forest   |    6  |   31   |   2   | ... |   ...  |
    |   urban    |    0  |    1   |  22   | ... |   ...  |
    |    ...     |  ...  |   ...  |  ...  | ... |   ...  |
    |------------+-------+--------+-------+-----|--------|
    |    sums    |  ...  |   ...  |  ...  | ... |   ...  |
```


### 5. Binary - cross matrix:
Confusion matrix for multi-class classification.
```
    |    | water | forest | ... |
    |----+-------+--------+-----|
    | TP |    1  |   55   | ... |
    | TN |   15  |   99   | ... |
    | FP |    5  |    3   | ... |
    | FN |   33  |   46   | ... |
```

 *where*:
 - columns: represent the classes in the dataset
 - rows: represent different types of classification outcomes for each class:
 - TP (True Positives): the number of samples correctly classified as a given class
 - TN (True Negatives): the number of samples that do not belong to a given class and were correctly identified as not belonging.
 - FP (False Positives): the number of samples incorrectly classified as a given class
 - FN (False Negatives): the number of samples of a given class that were incorrectly classified as not belonging to that class



# Metrics help

### 1. The definitions of the metrics are mainly based on the binary error matrix with the following symbols:
 - TP true positive
 - TN true negative
 - FP false positive
 - FN false negative.

### 2. Accuracy metrics classically used in remote sensing:
    - OA (overall_accuracy):
      OA = sum(TP) / (TP + TN + FP + FN)

    - PA (producer_accuracy):
      PA = TP / (TP + FN)

    - UA (user_accuracy)
      UA = TP / (TP + FP)

    - OME (omission errors / errors_of_omission):
      OME = FN / (TP + FN)

    - CME (errors_of_commision):
      CME = FP / (TP + FP)

    - NPV (negative predictive value):
      NPV = TN/(TN + FN) = 1 − FOR

### 3. Classification accuracy metrics found in contemporary scientific publications
 >some metrics overlap with some of the metrics mentioned in point 1.

These metrics can be conventionally divided into simple metrics (calculated directly from the TP, TN, FP and FN values) and
   complex metrics (calculated using simple metrics).

3.1. Simple metrics:

    - ACC (accuracy):
      ACC = (TP+TN)/(P+N) = (TP+TN)/(TP+TN+FP+FN)

    - PPV (precision or positive predictive value):
      PPV = TP / (TP + FP)

    - PPV (precision or positive predictive):
      PPV = TP / (TP + FP)

    - TPR (sensitivity, recall, hit rate, or true positive rate):
      TPR = TP/P = TP/(TP + FN) = 1 − FNR

    - TNR (specificity, selectivity or true negative rate):
      TNR = TN/N = TN/(TN + FP) = 1 − FPR

    - NPV (negative predictive value):
      NPV = TN/(TN + FN) = 1 − FOR

    - FNR (miss rate or false negative rate):
      FNR = FN/P = FN/(FN + TP) = 1 − TPR

    - FPR (fall-out or false positive rate):
      FPR = FP/N = FP/(FP + TN) = 1 − TNR

    - FDR (false discovery rate):
      FDR = FP/(FP + TP) = 1 − PPV

    - FOR (false omission rate):
      FOR = FN/(FN + TN) = 1 − NPV

    - TS / CSI (Threat score (TS) or critical success index (CSI)):
      TS = TP/(TP + FN + FP

    - MCC (Matthews correlation coefficient):
      mcc = (TP*TN - FP*FN) / [(TP+FP) * (TP+FN) * (TN+FP) * (TN+FN)]^0.5


3.2. Complex metrics:

    - PT (Prevalence Threshold):
      PT = ([TPR*(1 − TNR)]^0.5 + TNR − 1) / (TPR + TNR − 1)

    - BA (Balanced accuracy):
      ba = (TPR + TNR)/2

    - F1 score (is the harmonic mean of precision and sensitivity):
      f1 = 2*(PPV*TPR)/(PPV+TPR) = (2*TP)/(2*TP+FP+FN)

    - FM (Fowlkes–Mallows index):
      fm = [(TP/(TP+FP))*(TP/(TP+FN))]^0.5 = (PPV * TPR)^0.5

    - BM (informedness or Fowlkes–Mallows index):
      bm = TPR + TNR - 1

    - MK (markedness (MK) or deltaP):
      mk = PPV + NPV - 1

>$PT = ((TPR*(1 − TNR))^0.5 + TNR − 1) / (TPR + TNR − 1)$
