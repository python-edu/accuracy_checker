# 1. Skrypt `accuracy.py`

Skrypt  powstał w celu obliczania dokładności klasyfikacji różnych danych.

####  Katalog skryptu

> W katalogu skryptu powinny znajdować się pliki:

  - `accuracy.py` - skrypt
  - `gisEnv.yaml` - conda export środowiska uruchomieniowego skryptu
  - `raportForm` - formularz raportu


> W katalogu skryptu powinny znajdować się katalogi:

  - `jupyter` - zawiera pliki  `notebooki jupitera`:
    - gpkg2csv.ipynb
    - accuracy.ipynb

  - `testy` -  zawiera dane testowe i wyniki

## 2. Środowisko

Skrypt wymaga środowiska zawierającego pakiety wymienione w pliku `gisEnv.yaml`. Instalacja środowiska:

   > `conda env create -f gisEnv.yml`



## 3. Przygotowanie danych

Przykład przygotowania danych został przedstawiony w notatniku `gpkg2csv.ipynb`:
  > wydobycie danych z pliku wektorowego typu `gpkg` lub`shp`.



## 4. Wskaźniki dokładności

#### 4.1. Kod

Notatnik `accuracy.ipynb` zawiera kod użyty w skrypcie do obliczeń wskaźników dokładności. Nie w nim obliczonych wszystkich wskaźników lecz jest teoria i test kodu głównego.

#### 4.2. Definicje

Wskaźniki z dziedziny uczenia maszynowego opisane  są na stronie wikipedii [Sensitivity and specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)

---

## 5. Dane testowe i wyniki

Katalog `testy` zawiera:

 - dane testowe:
     - `dane01.gpkg` - plik wektorowy z danymi źródłowymi - do wydobycia (patrz `gpkg2csv.ipynb`)
     -  `dane01.csv` - dane wydobyte z pliku `dane01.gpkg` - dane wejściowe do   skryptu
     -  `cros01.csv` - cross matrix, alternatywne dane wejściowe do skryptu

 - formularz raportu - plik `raportForm.html`
 - wyniki działania skryptu:
     - `r01_cros.csv` - utworzoną przez skrypt lub będącą danymi wejściowymi confusion matrix
     - `r02_trueFalse.csv` - tabela operatorów klasyfikacji binarnej (TP, TN, FP, FN)
     - `r03_classicAcc.csv` - zestawienie klasycznych wskaźników dokładności
     - `r04_modern1.csv` i `r05_modern2.csv` - wskaźniki dokładności z maszynowego uczenia
     - `raport.html` - raport, zawierający wszystkie wyniki

---




