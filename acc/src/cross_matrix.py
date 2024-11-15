# --- ---*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd

# local imports

# ---

"""The module provides classes for:
    - calculating 'cross matrix' from data containing true and predicted
      labels.
    - if verbal names of classes are available, they are used
"""
# ---


class CrossMatrixValidation:
    """The class is used to check the correctness of the cross matrix read
    from the file. Such a cross can have different forms and various
    defects:
        - only numbers in the csv file
        - numbers and descriptions of rows and columns
        - missing some classes
        - classes with zero values (all zeros)
    """

    def __init__(self, data=None, etykieta="cl", scheme="normal", state="raw"):
        """
        Args:
          - data:  cross matrix, lista list, np.array lub pd.DataFrame
          - etykieta: str, etykieta do tworzenia nazw koumn i wierszy
                        np. cl --> cl_01, cl_02, ...

          - scheme: str, określa układ kolumn i wierszy:
                    - normal:  układ w wierszach `true` w kolumnach `predicted`
                    - reverse: układ odwrócony

          - state: str, określa zawartość cross matrix:
                   - raw:  same liczby, bez nazw kolumn i wierszy, bez sum
                   - cross: liczby z opisami kolumn/wierszy, bez sum
                   - full: wszystko, z sumami w wierszach i kolumnach

        Uwaga:
          - state: `raw` - dane to lista list lub np.array
          - state: `cross` lub `cross_full` - dane to pd.DataFrame!!!
        """

        self.data = copy.deepcopy(data)
        self._etykieta = etykieta
        self.scheme = scheme
        self.state = state
        self._labels = None
        self.map_labels = None
        self.cross_raw = None
        self.cross = None
        self.cross_full = None
    # --- ---

    def __call__(self, **kwargs):
        for key, value in kwargs.items():
            # Sprawdzamy, czy obiekt ma atrybut o nazwie key
            if hasattr(self, key):
                setattr(self, key, value)

        self._odpal()
    # --- ---

    def _odpal(self):
        """The method performs most of the calculations, so they can be
        triggered by other methods, e.g. when updating the value of an
        attribute.
        """

        # --- utwórz crosss tab w podstawowej wersji
        self._labels, self.cross_raw = self._make_cross_raw()

        # --- utwórz słownik mapujący true_values na labels
        self._labels, self.map_labels = self._make_labels_and_map()

        # --- wyrównaj liczbę wierszy i kolumn
        self.cross_raw = self._wyrownaj_cross()

        # --- usuń klasy z samymi zerami
        self.cross_raw = self._usun_puste()

        # --- zamień liczby na nazwy
        self.cross_raw = self._number_labels_to_strings()

        # --- wstaw opisy słowne do crosss
        self.cross = self._add_text_labels()

        # --- dodaj sumy wierszy i kolumn do crosss
        self.cross_full = self._add_sumy_rows_cols()

    # --- ---

    @staticmethod
    def _sort_dict(sl):
        return {k: v for k, v in sorted(sl.items(), key=lambda it: it[0])}

    # --- ---

    def _make_cross_raw(self, data=None, state=None, scheme=None, **kwargs):
        """Nadpisuje metodę klasy parent.
        Args:
          - data: cross matrix - lista list, np.array, pd.DataFrame.
                  Jeśli lista to każda pod lista to jeden wiersz macierzy.

        """
        for key, val in kwargs.items():
            locals()[key] = val

        if data is None:
            data = self.data
        if state is None:
            state = self.state
        if scheme is None:
            scheme = self.scheme

        # cross_raw to lista list / tupla
        if state == "raw":
            cross = np.array(data)
            labels = None

        # cross to: pd.DataFrame
        # liczby + opisy kolumn i wierszy, bez sum
        elif state == "cross":
            labels = tuple(data.columns)
            cross = data.to_numpy()

        # cross to: pd.DataFrame
        # liczby + opisy kolumn i wierszy + sumy
        elif state == "full":
            labels = tuple(data.columns)[:-1]
            cross = data.iloc[:-1, :-1].to_numpy()

        if self._labels is not None:
            labels = self.labels

        if scheme == "reverse":
            cross = cross.T

        cross = pd.DataFrame(cross)
        cross.index = list(range(1, cross.shape[0] + 1))
        cross.columns = list(range(1, cross.shape[1] + 1))
        cross.index.name = "true_values"
        cross.columns.name = "predicted"
        return labels, cross.astype("int")

    # --- ---

    def _make_labels_and_map(self):
        """If the input was just numbers (list of lists, e.g. array):
            - creates verbal descriptions of rows/columns based on the label,
              e.g. cl_1, cl_2

        If the input was pd.DataFrame, then they contained descriptions of
        rows and columns:
            - creates nothing

        Finally, creates `map_labels`: maps int numbers from 1 to verbal names
        i.e. labels.
        """

        etykieta = self.etykieta
        labels = self.labels
        n_rows = self.cross_raw.shape[0]

        # Tworzy mapę klas jeśli dane nie zawierają w sobie etykiet słownych
        if labels is None:
            if n_rows < 10:
                k = 1
            elif n_rows > 9 and n_rows < 100:
                k = 2
            elif n_rows > 99:
                k = 3

            labels = [f"{etykieta}_{i:0{k}}" for i in range(1, n_rows + 1)]

        klasy = [str(i) for i in range(1, len(labels) + 1)]
        map_labels = dict(set(zip(klasy, labels)))
        return tuple(labels), self._sort_dict(map_labels)

    # --- ---

    def _wyrownaj_cross(self):
        """If the classification results do not contain all true classes
        (e.g. some class was not detected), then the cross matrix has fewer
        columns (classification/prediction) than truth rows. You need to add
        columns with missing classes with zero occurrences.

        Args:
            - cross_raw: pd.crossstab, no row and column descriptions, no
              summaries.
        """
        cross = self.cross_raw.copy()
        s1 = set(cross.index.values)  # --- wartości prawdziwe - true
        s2 = set(cross.columns.values)  # --- kolumny crosss

        # --- uzupełnia kolumny
        if len(s1.difference(s2)) > 0:
            dif = s1.difference(s2)
            for n in dif:
                cross.loc[:, n] = [0 for i in range(cross.shape[0])]

        cross.sort_index(axis=0, inplace=True)
        cross.sort_index(axis=1, inplace=True)
        return cross.astype("int")

    # --- ---

    def _usun_puste(self):
        """Sometimes there are classes with all zeros, which causes Nan values
        and problems with division by zero. The function removes such rows and
        columns.

        Args:
            - cross_raw: pd.crossstab, no row and column descriptions,
              no summaries.
        """
        # jeśli dana klasa ma same zera to suma w wierszu i kolumnie jest
        # taka sama = 0. Numer wiersza i kolumny jest ten sam - macierz
        # kwadratowa.
        cross = self.cross_raw.copy()
        rows_sum = cross.sum(axis=1).to_numpy()
        idx = rows_sum == 0

        if idx.any():
            # rows index - do usunięcia
            ri = cross.index[idx]

            # cols index - do usunięcia
            ci = cross.columns[idx]

            cross.drop(index=ri, inplace=True)
            cross.drop(columns=ci, inplace=True)

        return cross
    # --- ---

    def _number_labels_to_strings(self):
        "When column and row labels are numbers - converts them to strings."
        cross = self.cross_raw.copy()
        kol_name = cross.columns.name
        row_name = cross.index.name

        kols = [str(x) for x in cross.columns]
        rows = [str(x) for x in cross.index]

        cross.columns = kols
        cross.columns.name = kol_name

        cross.index = rows
        cross.index.name = row_name
        return cross.astype("int")

    # --- ---

    def _add_sumy_rows_cols(self):
        cross = self.cross.copy()
        sum_row = cross.sum(axis=1).to_numpy()
        cross.loc[:, "sum_row"] = sum_row

        sum_col = cross.sum(axis=0).to_numpy()
        cross.loc["sum_col", :] = sum_col

        return cross.astype("int")

    # --- ---

    def _add_text_labels(self):
        """Creates a new cross matrix in which column and row names are
        changed from numbers to verbal descriptions, e.g. oats, rye, etc.
        """
        cross = self.cross_raw.copy()

        # --- pobierz aktualne liczbowe nazwy kolumn i wierszy
        all_names = cross.columns.to_list()

        if "sum_row" in all_names:
            old = all_names[:-1]
        else:
            old = all_names[:]
        # --- konwersja na string
        old = [str(x) for x in old]

        kols = [self.map_labels.get(key) for key in old]
        rows = kols[:]

        if len(rows) < len(all_names):
            kols.append("sum_row")
            rows.append("sum_col")

        cross.columns = kols
        cross.index = rows
        cross.axes[1].name = "predicted"
        cross.axes[0].name = "true"

        return cross.astype("int")

    # --- ---

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, values: list):
        self._labels = [str(a) for a in values]
        if self.cross_raw is not None:
            self._odpal()

    @property
    def etykieta(self):
        return self._etykieta

    @etykieta.setter
    def etykieta(self, value):
        self._etykieta = str(value)
        self._labels = None
        if self.cross_raw is not None:
            self._odpal()


# --- ---


class ConfusionMatrix:
    """
    # Result
      Returns cross matrix, in 3 three formats:
      1. 'cross_raw': raw result from `pd.crosstab()`, which:
            - has changed the names of column one and row one to
              `predicted` and `true_values`
            - no summarized rows and columns
            - no verbal descriptions of classes
            - equalized number of rows and columns
            - deleted rows and columns with all zeros
            - numbers in row and column names changed to `str`
            - table values changed to `int`

      1. 'cross_raw': surowy wynik z `pd.crosstab()`, który:
            - ma zmienione nazwy kolumny pierwszej i wiersza pierwszego na
              `predicted` i `true_values`
            - bez podsumowanych wierszy i kolumn
            - bez opisów słownych klas
            - wyrównana liczba wierszy i kolumn
            - usunięte wiersze i kolumny z samymi zerami
            - liczby w nazwach wierszy i kolumn zamienione na `str`
            - wartości tabeli zamienine na `int`

              predicted    1  2  3  4
              true
              1            3  0  1  0
              2            1  3  1  0
              3            0  2  2  0
              4            1  1  0  0


      2. 'cross': to samo co cross_raw ale z dodanymi opisami wierszy i kolumn.
            Opisy zależą od danych: albo są wzięte z pierwszej kolumny danych
            albo wygenerowane ze stałej etykiety z kolejnym numerem. Przykład
            poniżej zawiera domyślną etykietę klas.

              predicted  cl_1  cl_2  cl_3  cl_4
              true
              cl_1          3     0     1     0
              cl_2          1     3     1     0
              cl_3          0     2     2     0
              cl_4          1     1     0     0

      3. `cros_full`:  pełna macierz z sumami wierszy i kolumn

              predicted  cl_1  cl_2  cl_3  cl_4  sum_row
              true
              cl_1          3     0     1     0        4
              cl_2          1     3     1     0        5
              cl_3          0     2     2     0        4
              cl_4          1     1     0     0        2
              sum_col       5     6     4     0       15


    # Dane wejściowe / Input data
      Wyniki klasyfikacji obrazów w postaci tabeli o 2 lub 3 kolumnach. /
      Image classification results in the form of a table with 2 or 3 columns.

      1. Dwie kolumny / two columns:

             |  true_values | predicted |
             | ------------ | --------- |
             |     int      |    int    |
             |     ...      |    ...    |

      2. Trzy kolumny / three columns:

             | labels |  true_values | predicted |
             | ------ | ------------ | --------- |
             |   str  |     int      |    int    |
             |   ...  |     ...      |    ...    |

             gdzie labels to nazwy klas np. owies, przenica, woda, ...
    """

    def __init__(self, data=None, map_labels=None, etykieta="cl", scheme="normal"):
        """
        # Args:
          - data: 2 lub 3 kolumny w układzie:
            -- 3 kolumny: etykiety klas, true_values, predicted
            -- 2 kolumny: true_values, predicted
            Format danych: lista 2 lub 3 list, np.array, pd.DataFrame.

            Jeśli lista to każda kolumna to osobna lista tzn. np.:
            - [[etykiety klas], [true_values], [predicted]]

            Etykiety klas to słowne nazwy np. rzepak, woda, etc.

          - labels: dict {nr: name, ...} np. {1: pszenica, 2: zyto, ...}
          - etykieta:   str, jeśli nie ma nazw klas są tylko liczby 1,2,...
            to w nazwach kolumn i wierszy dodana będzie etykieta np.:
            klasy 1, 2, 3 --> 'cl_1', 'cl_2', 'cl_3'.

          - scheme: argument klasy parent
          - state: argument klasy parent
        """

        self._data = copy.deepcopy(data)
        self.etykieta = etykieta
        self.scheme = scheme
        self._labels = None
        self.map_labels = map_labels.copy() if map_labels else map_labels
        self.true_values = None
        self.predicted = None

        # inicjuje puste zmienne (jakby rejestruje je)
        self.cross = None
        self.cross_raw = None
        self.cross_full = None

        if data is not None:
            self.__call__()
    # --- ---

    def __call__(self):
        # breakpoint()
        self.map_labels, self._labels, self.true_values, self.predicted \
                = self._prepare_data()
        cross_raw = self._cross_raw_from_data()
        breakpoint()
        kwargs = {"_labels": self._labels}
        waliduj = CrossMatrixValidation(cross_raw,
                                        state="raw",
                                        scheme=self.scheme)
        waliduj(**kwargs)
        for atr in ["cross_raw", "cross", "cross_full"]:
            value = getattr(waliduj, atr)
            setattr(self, atr, value)
        # self._odpal()
    # --- ---

    # @staticmethod
    def _prepare_data(self):
        """Rozdziela wprowadzone dane (nie odczytuje ich z pliku!!!) na 3
        osobne np.array. Dane to wynik jakiejś klasyfikacji czyli tabela
        z wartościami i przydzielonymi etykietami klas. Czasem w tabeli może
        być 3 kolumna z ze słownymi etykietami klas np. woda, las, ...

        Args:
            - data: lista list np. ([[true_values, ...], [predicted, ...]],
              pd.DataFrame lub np.array. Dane to 2 lub 3
              kolumny:
              -- 2 kolumny: true_values, predicted
              -- 3 kolumny: labels, true_values, predicted
        """
        data = self.data
        
        if isinstance(data, list) or isinstance(data, tuple):
            data = np.array(data).T
        else:
            data = np.array(data)
        # breakpoint()
        if data.shape[1] == 2 and self.map_labels is None:
            cl = self.etykieta
            map1 = [(int(val), f'{cl}_{val}') for val in np.unique(data[:, 0])]
            map2 = [(int(val), f'{cl}_{val}') for val in np.unique(data[:, 1])]
            map_labels = dict(set(map1).union(map2))
            labels = [map_labels[key] for key in data[:, 0]]
        elif data.shape[1] == 3:
            # 2 pierwsze kolumny: labels, true_values
            # od pythona 3.7 dict -> unikalne wiersze z np.array
            tmp_dict = dict(data[:, :2])
            labels = tuple(tmp_dict.keys())

        true_values = data[:, -2].astype("int")
        predicted = data[:, -1].astype("int")

        return map_labels, labels, true_values, predicted

    # --- ---

    def _cross_raw_from_data(self):
        cros = pd.crosstab(
            self.true_values,
            self.predicted,
            dropna=False,
        )
        return cros.to_numpy()

    # --- ---

    @property
    def data(self):
        return copy.deepcopy(self._data)

    @data.setter
    def data(self, data):
        if data is not None:
            self.__call__(data)

    # --- ---
