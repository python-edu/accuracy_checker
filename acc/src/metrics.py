# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


"""
This module provides classes for calculating various accuracy metrics commonly
used in classification assessments, especially in remote sensing and machine
learning.

Classes:
  1. `AccClasic`:
      - Computes 'classic' metrics based on a confusion matrix (cross matrix).
  2. `AccClasicBin`:
      - Computes 'classic' metrics using a binary confusion matrix
        (binary cross matrix).
  3. `AccIndex`:
      - Computes accuracy metrics commonly used in machine learning.
"""


def handle_division_errors(func):
    """
    A decorator for handling division errors in metric calculations.

    Ensures division by zero and invalid operations (e.g., 0/0) are
    handled safely. Replaces such results with `np.nan`.

    Args:
        func: The function performing a division operation.

    Returns:
        The wrapped function with division error handling.
    """

    def wrapper(*args, **kwargs):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = func(*args, **kwargs)
            if isinstance(result, np.ndarray):  # Obsługa tablic NumPy
                result[np.isnan(result)] = np.nan
            elif isinstance(result, (float, int)):  # dla pojedynczych liczb
                if np.isnan(result):
                    result = np.nan
            return result
    return wrapper


class AccClasic:
    """
    Calculates 'classic' accuracy metrics from a cross matrix
    (confusion matrix).

    Input Data:
        - The input data should be a pandas DataFrame or a NumPy array.
        - A cross matrix without row and column summaries is expected.

    Example Input Cross Matrix:
        predicted |
          true    | water | forest | urban
       -----------+-------+--------+-------
        water     |   21  |    5   |   7
        forest    |    6  |   31   |   2
        urban     |    0  |    1   |  22

       - Rows represent reference (true) values.
       - Columns represent predicted values.


    Metrics:

      1. OA - overall accuracy:
                        OA = sum(all_good) / sum(all)

      2. PA - producer accuracy:
                        PA = sum(all_good) / sum(rows)

      3. UA -  user accuracy:
                        UA = sum(all_good) / sum(cols)

      4. OME - _errors_of_omission:
                        OME = (sum(rows) - sum(all_good)) / sum(rows)

      5. CME - errorsOfCommission
                        CME = (sum(cols) - sum(all_good)) / sum(cols)

    where:
      - `sum(all)` is the sum of all elements in the matrix.
      - `sum(all_good)` is the diagonal of the matrix (correct classifications)
      - `sum(rows)` is the sum of all values in a row.
      - `sum(cols)` is the sum of all values in a column.
    """

    # ---

    def __init__(self, data, precision=7, revers=False):
        """
        Args:
            data: A cross matrix (pandas DataFrame or NumPy array).
            precision: Number of decimal places for rounding results.
            revers: If True, reverses the matrix layout where:
                - Columns represent true values.
                - Rows represent predicted values.
        """
        self.precision = precision

        if revers:
            self.data = self._reverse_data(data)
            self.data = self._get_data(self.data)
        else:
            self.data = self._get_data(data)
        # breakpoint()
        self._calculate_sums()
        self._calculate_accuracy_metrics()

    # ---

    def _get_data(self, data):
        data = data.copy()
        if isinstance(data, pd.DataFrame):
            self.columns = data.columns.to_list()
            self.rows = data.index.to_list()
            data = data.to_numpy()
        return data

    def _reverse_data(self):
        return self.data.T

    # ---

    def _calculate_sums(self):
        self._total = self.data.sum()
        self._rows_sum = self.data.sum(axis=1)  # sumy w wierszach
        self._cols_sum = self.data.sum(axis=0)
        self._diagonalne = np.diagonal(self.data)  # wartości diagonalne

    # ---

    def _calculate_accuracy_metrics(self):
        self.OA = self._overall_accuracy()
        self.PA = self._producer_accuracy()
        self.UA = self._user_accuracy()

        self.OME = self._errors_of_omission()
        self.CME = self._errors_of_commision()

        self.tabela = self._table_results()
        self.tabela = self._round_table()

    # ---

    @handle_division_errors
    def _overall_accuracy(self):
        all_good = self._diagonalne.sum()
        return np.round(all_good / self._total, self.precision)

    @handle_division_errors
    def _errors_of_omission_old(self):
        diff = self._rows_sum - self._diagonalne
        return np.round(diff / self._rows_sum, self.precision)

    def _errors_of_omission(self):
        diff = self._rows_sum - self._diagonalne
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.true_divide(diff, self._rows_sum)
            # Zamień wartości wynikające z dzielenia przez zero na np.nan.
            result[np.isnan(result)] = np.nan
        return np.round(result, self.precision)

    @handle_division_errors
    def _errors_of_commision(self):
        diff = self._cols_sum - self._diagonalne
        return np.round(diff / self._cols_sum, self.precision)

    def _producer_accuracy(self):
        # return np.round(self._diagonalne / self._rows_sum, self.precision)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.true_divide(self._diagonalne,
                                    self._rows_sum,
                                    where=self._rows_sum != 0,
                                    out=np.full_like(self._diagonalne,
                                                     np.nan,
                                                     dtype=float)
                                    )
        return np.round(result, self.precision)

    @handle_division_errors
    def _user_accuracy(self):
        return np.round(self._diagonalne / self._cols_sum, self.precision)

    # ....................................................

    def _table_results(self):
        df = pd.DataFrame([])
        kols = ["OA", "PA", "UA", "OME", "CME"]
        # słownik atrybutów instancji
        sl = vars(self)
        # liczba wierszy zależna od liczby klas
        nrow = sl["PA"].size
        oa = [self.OA for i in range(nrow)]

        for k, v in sl.items():
            if k != "OA" and k in kols[1:]:
                df.loc[:, k] = v
        df.insert(0, "OA", oa)
        df = df.loc[:, kols]

        # przywróć nazwy wierszy
        df.index = self.rows
        return df

    def _round_table(self):
        tabela = self.tabela.copy()
        tabela = np.round(tabela, self.precision)
        return tabela


# =============================================================================


class AccClasicBin(AccClasic):
    """
    Oblicza te same metryki (klasyczne) co klasa `AccClasic` - różni się danymi
    wejściowymi - binary cross matrix.


    # Dane

    Dane wejściowe:   pd.DataFRame, tabela true/false w układzie:
                      ------------------------
                      |    |owies| zyto| ... |
                      |----+-----+-----+-----|
                      | TP |  1  |  55 | ... |
                      | TN | 15  |  99 | ... |
                      | FP |  5  |   3 | ... |
                      | FN | 33  |  46 | ... |
                      ------------------------

    # Metryki

      1. NPV (negative predictive value):
                NPV = TN/(TN + FN) = 1 − FOR

      2. OA (overall_accuracy):
                OV = suma(TP) / (TP + TN + FP + FN)

      3. PA (producer_accuracy):
                PA = TP / (TP + FN)

      4. UA (user_accuracy)
                UA = TP / (TP + FP)

      5. OME (omission errors / errors_of_omission):
                OME = FN / (TP + FN)

      6. CME (errors_of_commision):
                CME = FP / (TP + FP)

    """
    # ---

    def _get_data(self, data):
        return data.copy()

    @handle_division_errors
    def _x1npv(self):
        """negative predictive value (NPV)
        NPV = TN/(TN + FN) = 1 − FOR"""
        licznik = self.tf.loc["TN", :]
        mian = self.tf.loc[["TN", "FN"], :].sum(axis=0)
        return licznik / mian

    # ---

    @handle_division_errors
    def _overall_accuracy(self):
        """OV = suma(TP) / (TP + TN + FP + FN)"""
        licznik = self.data.loc["TP", :].to_numpy().sum()
        # self.licznik = licznik
        mian = self.data.loc[["TP", "TN", "FP", "FN"], :].sum(axis=0).iat[0]
        # self.mian = mian
        return licznik / mian

    @handle_division_errors
    def _producer_accuracy(self):
        """PA = TP / (TP + FN)"""
        licznik = self.data.loc["TP", :]
        mian = self.data.loc["TP", :] + self.data.loc["FN", :]
        return licznik / mian

    @handle_division_errors
    def _user_accuracy(self):
        """UA = TP / (TP + FP)"""
        licznik = self.data.loc["TP", :]
        mian = self.data.loc["TP", :] + self.data.loc["FP", :]
        return licznik / mian

    @handle_division_errors
    def _errors_of_omission(self):
        """OME = FN / (TP + FN)"""
        licznik = self.data.loc["FN", :]
        mian = self.data.loc["TP", :] + self.data.loc["FN", :]
        return licznik / mian

    @handle_division_errors
    def _errors_of_commision(self):
        """CME = FP / (TP + FP)"""
        licznik = self.data.loc["FP", :]
        mian = self.data.loc["TP", :] + self.data.loc["FP", :]
        return licznik / mian

    def _calculate_sums(self):
        pass

    def _table_results(self):
        df = pd.DataFrame([])
        kols = ["OA", "PA", "UA", "OME", "CME"]
        # słownik atrybutów instancji
        sl = vars(self)
        # liczba wierszy zależna od liczby klas
        nrow = sl["PA"].size
        oa = [self.OA for i in range(nrow)]

        for k, v in sl.items():
            if k != "OA" and k in kols[1:]:
                df.loc[:, k] = v
        df.insert(0, "OA", oa)
        df = df.loc[:, kols]

        return df


# ============================================================================

# Nowoczesne wskaźniki dokładności obliczne na podstawie tablicy 'true / false'
# Dane wejściowe to przygotowana przez funkcję 'trueFalseTable()':
#           - trueFalse - tablica true / false - pandas DataFRame
#
# Wskaźniki dokładności podzielono na dwie grupy: (1) proste i (2) złożone
#
#
# 1. Wskaźniki proste - obliczne są bezpośrednio z tabeli trueFalse czyli
#    używając TP, TN, FP, FN:
#   - acc, precision, sensitivity, specificity


class AccIndex:
    """
    Oblicza szereg wskaźników dokładności stosowanych w ocenie klasyfikacji
    szczególnie w 'machine learning'.


    # Dane

    Dane wejściowe:   pd.DataFRame, tabela true/false w układzie:

                      +----+-----+-----+-----+
                      |    |owies| żyto| ... |
                      +----+ --- + --- +---- +
                      | TP |  1  |  55 | ... |
                      | TN | 15  |  99 | ... |
                      | FP |  5  |   3 | ... |
                      | FN | 33  |  46 | ... |
                      +----+-----+-----+-----+


    # Podział metryk

    Metryki podzielono na 2 grupy:
      - proste, czyli obliczane używając wartości TP, TN, FP, FN. Metody
        obliczające te metryki poprzedzone są przedrostkiem `_x1`

      - złożone, obliczne z wykorzsytaniem innych metryk i wartości TP, TN, FP,
        FN. Metody obliczające te wskaźniki mają przedrostek: `_x2`.


    # Metryki

    ## Metryki proste

      1.  ACC (accuracy):
                ACC = (TP+TN)/(P+N) = (TP+TN)/(TP+TN+FP+FN)

      2. PPV (precision or positive predictive value):
                PPV = TP / (TP + FP)

      3. PPV (precision or positive predictive):
                PPV = TP / (TP + FP)

      4. TPR (sensitivity, recall, hit rate, or true positive rate):
                TPR = TP/P = TP/(TP + FN) = 1 − FNR

      5. TNR (specificity, selectivity or true negative rate):
                TNR = TN/N = TN/(TN + FP) = 1 − FPR

      6. NPV (negative predictive value):
                NPV = TN/(TN + FN) = 1 − FOR

      7. FNR (miss rate or false negative rate):
                FNR = FN/P = FN/(FN + TP) = 1 − TPR

      8. FPR (fall-out or false positive rate):
                FPR = FP/N = FP/(FP + TN) = 1 − TNR

      9. FDR (false discovery rate):
                FDR = FP/(FP + TP) = 1 − PPV

      10. FOR (false omission rate):
                FOR = FN/(FN + TN) = 1 − NPV

      11. TS / CSI (Threat score (TS) or critical success index (CSI)):
                TS = TP/(TP + FN + FP

      12. MCC (Matthews correlation coefficient):
        mcc = (TP*TN - FP*FN) / [(TP+FP) * (TP+FN) * (TN+FP) * (TN+FN)]^0.5


    ## Metryki złożone

      1 PT (Prevalence Threshold):
                PT = {[TPR*(1 − TNR)]^0.5 + TNR − 1} / (TPR + TNR − 1)

      2. BA (Balanced accuracy):
                ba = (TPR + TNR)/2

      3. F1 score (is the harmonic mean of precision and sensitivity):
                f1 = 2*(PPV*TPR)/(PPV+TPR) = (2*TP)/(2*TP+FP+FN)

      4. FM (Fowlkes–Mallows index):
                fm = [(TP/(TP+FP))*(TP/(TP+FN))]^0.5 = (PPV * TPR)^0.5

      5. BM (Bookmaker informedness):
                bm = TPR + TNR - 1

      6. MK (markedness (MK) or deltaP):
                mk = PPV + NPV - 1
    """

    def __init__(self, data, precision=7):
        self.tf = data.copy().astype(np.float64)
        self.precision = precision
        self._over_methods_x1()
        self.over_methods_x2()
        # for k,v in sorted(vars(self).items()):
        #   print(f'\n{k}:\n{v}\n')
        # print('\nvars(self):\n',vars(self))

    # ....................................................

    def _over_methods_x1(self):
        """Wykonuje metody obliczające indeksy na podstawie wartości TP, TN,
        FP, FN z tabeli 'binTF'. Wskaźniki te wcześniej były w grupie
        'modern1'.
        """
        for m in dir(AccIndex):
            # if re.search(r'^_{1}[a-z]+',m):
            if callable(getattr(AccIndex, m)) and m.startswith("_x1"):
                kod = (
                    f"""self.{m[4:]} = np.round(self.{m}(), self.precision)"""
                )
                # print(f'{m}\t{kod}')
                exec(kod)

    def over_methods_x2(self):
        for m in dir(AccIndex):
            if callable(getattr(AccIndex, m)) and m.startswith("_x2"):
                kod = (
                    f"""self.{m[4:]} = np.round(self.{m}(), self.precision)"""
                )
                # print(f'{m}\t{kod}')
                exec(kod)

    # ---

    @handle_division_errors
    def _x1_acc(self):
        """accuracy (ACC):
        ACC = (TP+TN)/(P+N) = (TP+TN)/(TP+TN+FP+FN)
        """
        licznik = self.tf.loc[["TP", "TN"], :].sum(axis=0)
        mian = self.tf.loc[["TP", "TN", "FP", "FN"], :].sum(axis=0)
        # print(f'\n\nSprrrrrr:\n\n{licznik}\n\n{mian}\n\nkonirc\n\n')
        return licznik / mian

    @handle_division_errors
    def _x1_ppv(self):
        """precision or positive predictive value (PPV)
        PPV = TP / (TP + FP)
        """
        licznik = self.tf.loc["TP", :]
        mian = self.tf.loc[["TP", "FP"], :].sum(axis=0)
        return licznik / mian

    @handle_division_errors
    def _x1_tpr(self):
        """sensitivity, recall, hit rate, or true positive rate (TPR)
        TPR = TP/P = TP/(TP + FN) = 1 − FNR"""
        licznik = self.tf.loc["TP", :]
        mian = self.tf.loc[["TP", "FN"], :].sum(axis=0)
        return licznik / mian

    @handle_division_errors
    def _x1_tnr(self):
        """specificity, selectivity or true negative rate (TNR)
        TNR = TN/N = TN/(TN + FP) = 1 − FPR"""
        licznik = self.tf.loc["TN", :]
        mian = self.tf.loc[["TN", "FP"], :].sum(axis=0)
        return licznik / mian

    @handle_division_errors
    def _x1_npv(self):
        """negative predictive value (NPV)
        NPV = TN/(TN + FN) = 1 − FOR"""
        licznik = self.tf.loc["TN", :]
        mian = self.tf.loc[["TN", "FN"], :].sum(axis=0)
        return licznik / mian

    @handle_division_errors
    def _x1_fnr(self):
        """miss rate or false negative rate (FNR)
        FNR = FN/P = FN/(FN + TP) = 1 − TPR"""
        licznik = self.tf.loc["FN", :]
        mian = self.tf.loc[["FN", "TP"], :].sum(axis=0)
        return licznik / mian

    @handle_division_errors
    def _x1_fpr(self):
        """fall-out or false positive rate (FPR)
        FPR = FP/N = FP/(FP + TN) = 1 − TNR"""
        licznik = self.tf.loc["FP", :]
        mian = self.tf.loc[["FP", "TN"], :].sum(axis=0)
        return licznik / mian

    @handle_division_errors
    def _x1_fdr(self):
        """false discovery rate (FDR)
        FDR = FP/(FP + TP) = 1 − PPV"""
        licznik = self.tf.loc["FP", :]
        mian = self.tf.loc[["FP", "TP"], :].sum(axis=0)
        return licznik / mian

    @handle_division_errors
    def _x1_foRate(self):
        """false omission rate (FOR)
        FOR = FN/(FN + TN) = 1 − NPV"""
        licznik = self.tf.loc["FN", :]
        mian = self.tf.loc[["FN", "TN"], :].sum(axis=0)
        return licznik / mian

    @handle_division_errors
    def _x1_ts(self):
        """Threat score (TS) or critical success index (CSI)
        TS = TP/(TP + FN + FP)"""
        licznik = self.tf.loc["TP", :]
        mian = self.tf.loc[["TP", "FN", "FP"], :].sum(axis=0)
        return licznik / mian

    @handle_division_errors
    def _x1_mcc(self):
        """Matthews correlation coefficient (MCC)
        mcc = (TP*TN - FP*FN) / [(TP+FP) * (TP+FN) * (TN+FP) * (TN+FN)]^0.5
        """
        tp = self.tf.loc["TP", :]
        tn = self.tf.loc["TN", :]
        fp = self.tf.loc["FP", :]
        fn = self.tf.loc["FN", :]

        licznik = (tp * tn) - (fp * fn)
        mian = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        return licznik / mian

    # -----------------------------------------------------------------
    # wskaźniki złożone - modern2. Są obliczne na podstawie
    # wskaźników z grupy 1 - modern1.

    @handle_division_errors
    def _x2_pt(self):
        """Prevalence Threshold (PT)
        PT = {[TPR*(1 − TNR)]^0.5 + TNR − 1} / (TPR + TNR − 1)"""
        licznik = (self.tpr * (1 - self.tnr)) ** 0.5 + (self.tnr - 1)
        # mian = self.tf.loc[:,['tpr','tnr']].sum(axis=1)-1
        mian = self.tpr + self.tnr - 1
        return licznik / mian

    def _x2_ba(self):
        """Balanced accuracy (BA):
        ba = (TPR + TNR)/2
        """
        return (self.tpr + self.tnr) / 2

    @handle_division_errors
    def _x2_f1(self):
        """F1 score is the harmonic mean of precision and sensitivity
        f1 = 2*(PPV*TPR)/(PPV+TPR) = (2*TP)/(2*TP+FP+FN)
        """
        licznik = 2 * self.ppv * self.tpr
        mian = self.ppv + self.tpr
        return licznik / mian

    def _x2_fm(self):
        """Fowlkes–Mallows index (FM)
        fm = [(TP/(TP+FP))*(TP/(TP+FN))]^0.5 = (PPV * TPR)^0.5
        """
        return (self.ppv * self.tpr) ** 0.5

    def _x2_bm(self):
        """Bookmaker informedness (BM)
        bm = TPR + TNR - 1
        """
        return self.tpr + self.tnr - 1

    def _x2_mk(self):
        """markedness (MK) or deltaP
        mk = PPV + NPV - 1
        """
        return self.ppv + self.npv - 1


class CustomMetrics1:
    def __init__(self, binary_cross, formula, precision=4):
        self.binary_cross = binary_cross.copy()
        self.formula = formula
        self.precision = precision
        self.results = self._calculate_metric()

    def _prepare_forrmula(self):
        variable, formula = [x.strip() for x in self.formula.split("=")]
        return variable, formula

    def _calculate_metric(self):
        """
        """
        results = {}
        df = self.binary_cross.copy()
        variable, formula = self._prepare_forrmula()

        for column in df.columns:
            # Kontekst dla eval
            context = {key: df.loc[key, column] for key in df.index}
            context = {key: int(val) for key, val in context.items()}
            try:
                # Sprawdzenie, czy denominator w formule jest zerem
                if "/" in formula:
                    denominators = [
                        eval(den, {"__builtins__": None}, context)
                        for den in formula.split("/")[-1:]
                    ]
                    if any(den == 0 for den in denominators):
                        results[column] = float("nan")
                        continue

                results[column] = eval(formula,
                                       {"__builtins__": None},
                                       context
                                       )
            except Exception as e:
                results[column] = f"Error: {e}"  # Informacja o błędzie

        df = pd.DataFrame(results, index=[variable])
        df = np.round(df, self.precision)
        return df


class CustomMetrics:
    """
    A class to compute custom metrics for binary classification results based
    on a provided formula.

    Attributes:
        binary_cross (pd.DataFrame): DataFrame containing binary classification
                                     results.
        formula (str): A string defining the metric formula in the format
                      'variable = expression'.
        precision (int): Decimal precision for the results. Defaults to 4.
        results (pd.DataFrame): Computed metric results.
    """

    def __init__(self, binary_cross, formula, precision=4):
        """
        Initialize the CustomMetrics class.

        Args:
            binary_cross (pd.DataFrame): DataFrame containing binary
                         classification results.
            formula (str): Metric formula in the format 'variable = expression'
            precision (int, optional): Decimal precision for the results.
                         Defaults to 4.
        """
        self.binary_cross = binary_cross.copy()
        self.formula = formula
        self.precision = precision
        self.results = self._calculate_metric()

    def _prepare_formula(self):
        """
        Parse the metric formula into a variable name and an evaluable
        expression.

        Returns:
            tuple: A tuple containing the variable name (str) and the formula
                   (str).
        """
        formula = self.formula.upper()
        splited = [x.strip() for x in formula.split("=")]
        if len(splited) == 2:
            metric, formula = splited
        else:
            formula = splited[0]
            metric = 'custom'
        return metric, formula

    def _calculate_metric(self):
        """
        Compute the custom metric for each column in the binary_cross
        DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the computed metric values
                          for each column.
        """
        results = {}
        df = self.binary_cross.copy()
        variable, formula = self._prepare_formula()

        for column in df.columns:
            # Create a context dictionary for evaluating the formula
            context = {key: df.loc[key, column] for key in df.index}
            context = {key: int(val) for key, val in context.items()}
            try:
                # Check for division by zero in the formula
                if "/" in formula:
                    denominators = [
                        eval(den, {"__builtins__": None}, context)
                        for den in formula.split("/")[-1:]
                    ]
                    if any(den == 0 for den in denominators):
                        # Assign NaN for division by zero errors
                        results[column] = float("nan")
                        continue

                results[column] = eval(formula,
                                       {"__builtins__": None},
                                       context
                                       )
            except Exception as e:
                # Capture and store any evaluation errors
                results[column] = f"Error: {e}"

        # Convert results to a DataFrame and round to the specified precision
        df = pd.DataFrame(results, index=[variable])
        df = np.round(df, self.precision)
        return df
# ---


if __name__ == "__main__":
    pass
