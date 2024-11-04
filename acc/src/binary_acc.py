# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# ---


wersja = "w2.2022.03.30"
opis = """
Wersja modułu: {0}.

Klasa tworzy table wskaźników TF, TN, FP, FN zwaną 'bin_tab' na podstawie
'cross matrix.

Uwaga!!!!

    CrossMatrix nie może zawierać podsumowania kolumn i wierszy!!!!

Przykład użycia:

    bt = BinTable()
    cros =
              water  forest  urban
    water      21       5      7
    forest      6      31      2
    urban       0       1     22

    bt(cros)

          water  forest  urban
    TP     21      31     22
    TN     56      50     63
    FP      6       6      9
    FN     12       8      1

""".format(
    wersja
)


# -----------------------------------------------------------------------------

class BinTable:
    def __init__(self,
                 data_frame=None,
                 layout="h",
                 row_names=("TP", "TN", "FP", "FN")
                 ):
        """
        Args:
            - data:     pd.DataFrame, ConfusionMatrix
            - layout:   str, h-horyontalnie, v-vertykalnie, określa układ
                        tabeli wynikowej
        """
        self.layout = layout
        self.row_names = row_names  # ("TP", "TN", "FP", "FN")

        if data_frame is not None:
            self.__call__(data_frame, row_names)
            self.data = data_frame
        # if self.layout == "v":
        #    self.binTF = self.binTF.T

    # -----------------------------------------------------------------
    def __call__(self, data_frame, row_names=None):
        """Tworzy binarną tabelę wskaźników.
        Args:
        - data_frame: cross matrix, pd.DataFrame
        Returns:
        -
        """
        if row_names is None:
            row_names = self.row_names

        ar, col_names = self._get_data(data_frame)
        bin_tab = self._bin_table(ar, col_names, row_names)
        if self.layout == "v":
            bin_tab = bin_tab.T
        return bin_tab

    # -----------------------------------------------------------------

    def _get_data(self, data_frame):
        """Zmienia DataFrame na numpy array i listę nazw kolumn."""
        data = data_frame.copy()
        return data.to_numpy(), tuple(data.columns.to_list())

    # -----------------------------------------------------------------

    def _bin_table(self, array, col_names: list, row_names: list):
        """Tworzy binary table"""
        ar = array
        cols = col_names
        sl = {}
        # rowsIdx = ['TP','TN','FP','FN']

        for i in range(ar.shape[0]):
            tp = ar[i, i]

            tmp = np.delete(ar, i, 1)  # usuń kolumnę rozpatrywanej klasy
            tmp = np.delete(tmp, i, 0)  # usuń wiersz rozpatrywanej klasy

            tn = tmp.sum()

            # pobierz wiersz i usuń z niego rozpatrywaną kolumnę
            row = np.delete(ar[i, :], i)
            fn = row.sum()

            # pobierz kolumnę i usuń z niej rozpatrywany wiersz
            col = np.delete(ar[:, i], i)
            fp = col.sum()

            sl[cols[i]] = [tp, tn, fp, fn]

        wyn = pd.DataFrame(sl, index=row_names)
        # wyn.loc['total',:] = wyn.sum(axis=0)
        return wyn.astype("int")


# ---
