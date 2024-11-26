# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# ---

"""
Moduł zawiera klasy dostarczające metody do obliczania różnych wskaźników
dokładności.

Klasy:
  ## 1. 'AccClasic'    - oblicza wskaźniki 'klasyczne' na podstawie
                         'cross matrix'
  ## 2. 'AccClasicBin' - oblicza wskaźniki 'klasyczne' na podstawie 'binTF'
  ## 3. 'AccIndex'     - oblicza wskaźniki z machine learning na podstawie
                         'binTF'

"""

def handle_division_errors(func):
    def wrapper(*args, **kwargs):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = func(*args, **kwargs)
            if isinstance(result, np.ndarray):  # Obsługa tablic NumPy
                result[np.isnan(result)] = np.nan
            elif isinstance(result, (float, int)):  # Obsługa pojedynczych liczb
                if np.isnan(result):
                    result = np.nan
            return result
    return wrapper

# -----------------------------------------------------------------------------


class AccClasic:
    """\
    # Dane

        predicted |                     
          true    | water | forest | urban
       -----------+-------+--------+-------
        water     |   21  |    5   |   7
        forest    |    6  |   31   |   2
        urban     |    0  |    1   |  22

       - wartości referencyjne w wierszach
       - wartości `predicted` w kolumnach

      Dane wejściowe:
        - pd.DataFRame lub np.array
        - cross matrix - bez podsumowań wierszy i kolumn!!!!


    # Metryki

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

    gdzie:
      - sum(all):       suma wszytskich komórek w cross matrix
      - sum(all_good):  suma wartości na przekątnej cross matrix
      - sum(rows):      suma wierszy
      - sum(cols):      suma kolumn

    """

    # ---

    def __init__(self, data, precision=7, revers=False):
        """
        Args:
          - data: cross matrix, pd.DataFrame lub np.array
          - revers: wskazuje, że jest odwrócony układ cross matrix:
                    * kolumny to prawda
                    * wiersze to predicted
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
            # Zamień wartości wynikające z dzielenia przez zero na 0 (lub np.nan).
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
            # Zamień wartości wynikające z dzielenia przez zero na 0 (lub np.nan).
            # result[np.isnan(result)] = -1
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

      5. BM (informedness or Fowlkes–Mallows index):
                bm = TPR + TNR - 1

      6. MK (markedness (MK) or deltaP):
                mk = PPV + NPV - 1
    """

    def __init__(self, data, precision=7):
        self.tf = data.copy().astype(np.float128)
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
        """informedness or Fowlkes–Mallows index (BM)
        bm = TPR + TNR - 1
        """
        return self.tpr + self.tnr - 1

    def _x2_mk(self):
        """markedness (MK) or deltaP
        mk = PPV + NPV - 1
        """
        return self.ppv + self.npv - 1


###############################################################################

if __name__ == "__main__":
    pass
