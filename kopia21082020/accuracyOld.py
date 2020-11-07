# -*- coding: utf-8 -*-

wersja = 'w1.2020.06.30'
opis = '''
Wersja skryptu: {0}.

Skrypt sprawdza jakość klasyfikacji:
   - tworzy lub wykorzystuje istniejącą cross matrix
   - oblicza metryki dokładności.

Metryki proste:
   - acc:  accuracy
   - ppv:  precision or positive predictive value
   - tpr:  sensitivity, recall, hit rate, true positive rate
   - tnr:  specificity, selectivity or true negative rate
   - npv:  negative predictive value
   - fnr:  miss rate or false negative rate 
   - fpr:  fall-out or false positive rate 
   - fdr:  false discovery rate
   - foRate: false omission rate
   - ts:  Threat score (TS) or critical success index (CSI)
   - mcc: Matthews correlation coefficient, od -1 do 1


Metryki złozone:
   - pt:  prevalence threshold
   - ba:  balanced accuracy
   - f1:  harmonic mean of precision and sensitivity
   - fm:  Fowlkes–Mallows index
   - bm:  Fowlkes–Mallows index
   - mk:  markedness or deltaP

'''.format(wersja)


import sys
import pandas as pd

sys.path.append('/home/u1/03_Programowanie/03Python/skrypty/skryptyCht2@agh/generals/')
#sys.path.append('/home/u1/anaconda3/envs/gis/lib/python3.8/site-packages/pandas')

import os, json, jinja2
import argparse    # moduł zalecany w pythonie do parsowania argumentów lini poleceń
import textwrap
from printing import printDict, printList

from pathlib import Path

import numpy as np
from tabulate import tabulate



#########################################################################################################

def parsujArgumenty():
    '''
    '''
    
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter,description=opis)
    
#kols=['nazwa','true','predicted']    
    parser.add_argument('input',  type=str,   help=textwrap.fill(f'''Adres pliku 'csv' z danymi. Dane to:
                                                                1. Surowe dane - przynajmniej trzy kolumny:
                                                                ----------------------------    
                                                                | nazwa | true | predicted |
                                                                | ----- | ---- | --------- |
                                                                | owies |   1  |     2     |
                                                                | trawa |   5  |     3     |
                                                                ----------------------------
                                                                :
                                                                    gdzie:
                                                                - 'nazwa' - skrócone nazwy roślin np. owies,
                                                                - 'predict' np. 5 - wynik klasyfikacji
                                                                - 'true' np. 7 - prawdziwa etykieta klasy.
                                                                
                                                                2. Cross matrix - gotowa tabela z opisami i sumami. Wymaga podania argumentu opcjonalnego '-c'.''',width = 70))
    
    parser.add_argument('-c','--cross',   help=textwrap.fill(u"Wskazuje, że dane wejściowe to cross matrix.", width = 100), action = 'store_true')

    parser.add_argument('-r','--revers',   help=textwrap.fill('''Wskazuje, że układ cross matrix jest odwrócony
                                                            tzn. w kolumnach są dane referencyjne a w wierszach 
                                                            wyniki klasyfikacji (predict).''', width = 100), action = 'store_true')

    parser.add_argument('-s','--save',   help=textwrap.fill('''Powoduje, że wyniki zapisane zostaną do osobnych plików
                                                            csv: cros.csv, trueFalse.csv, classicAcc.csv, modern1.csv,
                                                            modern2.csv.''', width = 100), action = 'store_true')

    parser.add_argument('-rap','--raport',   help=textwrap.fill('''Generuje raport w html, czyli wszytskie tabele
                                                            w jednym pliku html - raport.html''', width = 100), action = 'store_true')
    
    
    parser.add_argument('-v','--verbose',   help=textwrap.fill(u"Wyświetla bardziej szczegółowe informacje.", width = 100), action = 'store_true')
    
    args = parser.parse_args() 
    return args


#-----------------------------------------------------------------------------------------------------------------------

def przygotujDaneInput(args):
    '''Funkcja przetwarza argumenty wejściowe skryptu'''
    
    # ustal katalog z którego uruchamiany jest skrypt
    
    args.runDir = Path(__file__).resolve().parent.as_posix()
    
    if args.input:
        args.input = Path(args.input).resolve()
        args.out = args.input.with_name('cross.csv').resolve().as_posix()
        args.workDir = args.input.parent.as_posix()
        args.input = args.input.as_posix()
    
    # tworzy adresy do zapisu wyników
    if args.save:
        save = ['cros', 'trueFalse', 'classicAcc', 'modern1','modern2']
        sl ={}
        tmpName = Path(args.input).name.split('.')[0]
        for i,s in enumerate(save,1):
            name = f'r0{tmpName}_{i}_{s}.csv'
            sl[s] = Path(args.input).with_name(name).resolve().as_posix()
        args.save = sl
   
    if args.raport:
        name = Path(args.input).name.split('.')[0]
        name = f'{name}_raport.html'
        args.raport = Path(args.input).with_name(name).as_posix()
     
    
    return args



#-----------------------------------------------------------------------------------------------------------------------

def crossT(data):
    ''' Funkcja tworzy confiusion matrix wykorzystująć 'pd.crosstab()'.
        Args:
            - data:    pd.DataFRame, obowiązkowe 3 kolumny: short, true, predicted
'
        Out:
            - wynik:  pd.DataFrame
    '''
    df = data.copy()
    ref = ref[ref.loc[:,'short'] != 'noData']
    ref = ref.sort_values('true')
    
    
    # cross matrix
    cros = pd.crosstab(df.true,df.predicted,rownames=['true'],colnames=['predicted'])

    # Waliduj - uzupełnij kolumny / wiersze do cross matrix
    cros, noFillVal = walidujCros(cros,ref)
    
    # mapuj nazwy kolumn i wierszy
    if noFillVal:
        idx = ref.index[ref.short == 'fillVal'][0]
        ref.drop(idx,axis=0,inplace=True)
    cros.columns = ref.loc[:,'short']
    cros.index = ref.loc[:,'short']
    cros.axes[1].name='predicted'
    cros.axes[0].name='true'
    
    # dodaje sumy wierszy i kolumn
    sum1 = cros.sum(axis=0)
    cros.loc['sumKol',:] = sum1
    
    sum2 = cros.sum(axis=1)
    cros.loc[:,'sumRow'] = sum2
    
    return cros.astype('int')


#-----------------------------------------------------------------------------------------------------------------------

def walidujCros(cros,ref):
    ''' Jeśli wyniki klasyfikacji nie zawierają wszystkich prawdziwych klas (jakiejś klasy nie wykryto),
        to cross matrix ma mniej kolumn(klasyfikacja) niż wierszy prawdy. Trzeba dodać kolumny z brakującymi
        klasami z zerowymi wystąpieniami. Podobnie jest z wierszami.
        
        Jeśli nie występują brakujące wartości to kolumna/wiersz 'fillVal' są zbędne (same zera) - usuń je.
    '''
    cros,ref = cros.copy(),ref.copy()
    
    s1 = set(ref.true.values) # wartości prawdziwe z pliku referencjego
    s2 = set(cros.columns.values) # kolumny cross
    s3 = set(cros.index.values) # wiersze cross
    #print(f'\ncros: {list(s1)} / {s2}\nref: {list(s2)}\n')
    
    # uzupełnia kolumny
    if len(s1.difference(s2)) > 0:
        dif = s1.difference(s2)
        for n in dif:
            cros.loc[:,n] = [0 for i in range(cros.shape[0])]
            
        kols = list(cros.columns.values)
        kols.sort()
        cros = cros.loc[:,kols]
    
        #print(f'\n\ntuuu:\n{cros}\n\n')
     
     # uzupełnia wiersze
    if len(s1.difference(s3)) > 0:
        dif = s1.difference(s3)
        for n in dif:
            cros.loc[n,:] = [0 for i in range(cros.shape[1])]
            
        rows = list(cros.index.values)
        rows.sort()
        cros = cros.loc[rows,:]
    
    #print(f'\n\ntuuu:\n{cros}\n\n')
    
    # Jeśli nie występują brakujące wartości to kolumna/wiersz 'fillVal' są zbędne (same zera) - usuń je
    # pobierz id 'fillVal' - tak nazywa się wiersz i kolumna z fillVal
    fillVal = ref.true[ref.short == 'fillVal'].iat[0]
    
    if cros.loc[fillVal,:].sum() == 0 and cros.loc[:,fillVal].sum() == 0:
        cros.drop(fillVal,axis=0,inplace=True)
        cros.drop(fillVal,axis=1,inplace=True)
        noFillVal = 1
    else:
        noFillVal = 0
   
    return cros, noFillVal

    

#-----------------------------------------------------------------------------------------------------------------------

def trueFalseTable(cros):
    ''' f
        Args:
            - cros:    pd.DataFrame, cross matrix z wierszem i kolumną podsumowań
    '''
    ar = cros.copy()
    cols = ar.columns
    ar = ar.iloc[:-1,:-1].to_numpy() # pomiń wiersze i kolumny podsumowań
    k = ar.shape[0]
    sl = {}
    rowsIdx = ['TP','TN','FP','FN']
    
    for i in range(k):
        tp = ar[i,i]
        
        tmp = np.delete(ar,i,1)  # usuń kolumnę rozpatrywanej klasy
        tmp = np.delete(tmp,i,0) # usuń wiersz rozpatrywanej klasy

        tn = tmp.sum()
        
        row = np.delete(ar[i,:],i) # pobierz wiersz i usuń z niego rozpatrywaną kolumnę
        fn = row.sum()
        
        col = np.delete(ar[:,i],i) # pobierz kolumnę i usuń z niej rozpatrywany wiersz
        fp = col.sum()
        
        sl[cols[i]] = [tp,tn,fp,fn]
    
    wyn = pd.DataFrame(sl,index=rowsIdx)
    wyn.loc['total',:] = wyn.sum(axis=0)
    return wyn.astype('int')

# ----------------------------------------------------------------------------------------------------------------------

# precision dla standardowych wskaźników
prec = 2

def overallAccuracy(cros):
    total = cros.iloc[-1,-1] # ostatnia komórka (prawy dolny róg)
    allGood = np.trace(cros.iloc[:-1,:-1]) # suma liczb na przekątnej
    oacc = np.round((allGood / total)*100,prec)
    return oacc

def errorsOfOmission(cros):
    diag = np.diagonal(cros.iloc[:-1,:-1]) # wartości diagonalne
    rowsum = np.sum(cros.iloc[:-1,:-1],axis=1) # sumy w wierszach
    dif = rowsum - diag
    erOm = np.round((dif/rowsum)*100,prec)
    return erOm

def errorsOfCommission(cros):
    diag = np.diagonal(cros.iloc[:-1,:-1]) # wartości diagonalne
    kolsum = np.sum(cros.iloc[:-1,:-1],axis=0) # sumy w kolumnach
    dif = kolsum - diag
    erCom = np.round((dif/kolsum)*100,prec)
    return erCom

def producerAcc(cros):
    diag = np.diagonal(cros.iloc[:-1,:-1]) # wartości diagonalne
    rowsum = np.sum(cros.iloc[:-1,:-1],axis=1) # sumy w wierszach
    producer = np.round((diag/rowsum)*100,prec)
    return producer

def userAcc(cros):
    diag = np.diagonal(cros.iloc[:-1,:-1]) # wartości diagonalne
    kolsum = np.sum(cros.iloc[:-1,:-1],axis=0) # sumy w kolumnach
    user = np.round((diag/kolsum)*100,prec)    
    return user

# ================================================================================

# Nowoczesne wskaźniki dokładności obliczne na podstawie tablicy 'true / false'
# Dane wejściowe to przygotowana przez funkcję 'trueFalseTable()':
#           - trueFalse - tablica true / false - pandas DataFRame
#
# Wskaźniki dokładności podzielono na dwie grupy: (1) proste i (2) złożone
#
#
# 1. Wskaźniki proste - obliczne są bezpośrednio z tabeli trueFalse czyli używając: TP, TN, FP, FN
#   - acc, precision, sensitivity, specificity

factors1 = {'acc':'accuracy', 'ppv':'precision', 'tpr':'sensitivity', 'tnr':'specificity',\
            'npv': 'negative predictive value', 'fnr':'false negative rate',\
            'fpr':'false positive rate', 'fdr':'false discovery rate', 'foRate':'false omission rate',\
            'ts':'threat score', 'mcc': 'Matthews correlation coefficient od -1 do 1' }

def acc(trueFalse):
    ''' accuracy (ACC):
        ACC = (TP+TN)/(P+N) = (TP+TN)/(TP+TN+FP+FN)
    '''
    licznik = trueFalse.loc[['TP','TN'],:].sum(axis=0)
    mian = trueFalse.loc[['TP','TN','FP','FN'],:].sum(axis=0)
    #print(f'\n\nSprrrrrr:\n\n{licznik}\n\n{mian}\n\nkonirc\n\n')
    return licznik/mian


def ppv(trueFalse):
    ''' precision or positive predictive value (PPV)
        PPV = TP / (TP + FP)
    '''
    licznik = trueFalse.loc['TP',:]
    mian = trueFalse.loc[['TP','FP'],:].sum(axis=0)
    return licznik/mian


def tpr(trueFalse):
    ''' sensitivity, recall, hit rate, or true positive rate (TPR)
        TPR = TP/P = TP/(TP + FN) = 1 − FNR '''
    licznik = trueFalse.loc['TP',:]
    mian = trueFalse.loc[['TP','FN'],:].sum(axis=0)
    return licznik/mian


def tnr(trueFalse):
    ''' specificity, selectivity or true negative rate (TNR)
        TNR = TN/N = TN/(TN + FP) = 1 − FPR'''
    licznik = trueFalse.loc['TN',:]
    mian = trueFalse.loc[['TN','FP'],:].sum(axis=0)
    return licznik/mian


def npv(trueFalse):
    ''' negative predictive value (NPV)
        NPV = TN/(TN + FN) = 1 − FOR'''
    licznik = trueFalse.loc['TN',:]
    mian = trueFalse.loc[['TN','FN'],:].sum(axis=0)
    return licznik/mian


def fnr(trueFalse):
    ''' miss rate or false negative rate (FNR)
        FNR = FN/P = FN/(FN + TP) = 1 − TPR'''
    licznik = trueFalse.loc['FN',:]
    mian = trueFalse.loc[['FN','TP'],:].sum(axis=0)
    return licznik/mian


def fpr(trueFalse):
    ''' fall-out or false positive rate (FPR)
        FPR = FP/N = FP/(FP + TN) = 1 − TNR'''
    licznik = trueFalse.loc['FP',:]
    mian = trueFalse.loc[['FP','TN'],:].sum(axis=0)  
    return licznik/mian


def fdr(trueFalse):
    ''' false discovery rate (FDR)
        FDR = FP/(FP + TP) = 1 − PPV '''
    licznik = trueFalse.loc['FP',:]
    mian = trueFalse.loc[['FP','TP'],:].sum(axis=0)  
    return licznik/mian


def foRate(trueFalse):
    ''' false omission rate (FOR)
        FOR = FN/(FN + TN) = 1 − NPV '''
    licznik = trueFalse.loc['FN',:]
    mian = trueFalse.loc[['FN','TN'],:].sum(axis=0)
    return licznik/mian


def ts(trueFalse):
    ''' Threat score (TS) or critical success index (CSI)
        TS = TP/(TP + FN + FP) '''
    licznik = trueFalse.loc['TP',:]
    mian = trueFalse.loc[['TP','FN','FP'],:].sum(axis=0)
    return licznik/mian


def mcc(trueFalse):
    ''' Matthews correlation coefficient (MCC)
        mcc = (TP*TN - FP*FN) / [(TP+FP) * (TP+FN) * (TN+FP) * (TN+FN)]^0.5
    '''
    tf = trueFalse.copy()
    
    tp = tf.loc['TP',:]
    tn = tf.loc['TN',:]
    fp = tf.loc['FP',:]
    fn = tf.loc['FN',:]
    
    licznik = (tp * tn) - (fp * fn)
    mian = ((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))**0.5
    return licznik/mian


#
#
# 2. Wskażniki złożone - obliczane na podstawie wskaźników prostych
#   - 

factors2 = {'pt':'prevalence threshold', 'ba':'balanced accuracy', 'f1':'harmonic mean of precision and sensitivity',\
            'fm':'Fowlkes–Mallows index',\
            'bm':'Fowlkes–Mallows index', 'mk':'markedness or deltaP'}


def pt(factors):
    ''' Prevalence Threshold (PT)
        PT = {[TPR*(1 − TNR)]^0.5 + TNR − 1} / (TPR + TNR − 1) '''
    factors = factors.copy()
    #print(f'factors:\n{factors}\n\n')
    licznik1 = (factors.loc[:,'tpr'] * (1 - factors.loc[:,'tnr']))**0.5
    #print(f'licznik1:\n{licznik1}\n\n')
    licznik2 = factors.loc[:,'tnr'] - 1
    #print(f'licznik2:\n{licznik2}\n\n')
    licznik = licznik1 + licznik2
    #print(f'licznik:\n{licznik}\n\n')
    mian = factors.loc[:,['tpr','tnr']].sum(axis=1)-1
    return licznik/mian


def ba(factors):
    ''' Balanced accuracy (BA):
        ba = (TPR + TNR)/2
    '''
    factors = factors.copy()
    licznik = factors.loc[:,['tpr','tnr']].sum(axis=1)
    mian = 2
    return licznik/mian


def f1(factors):
    ''' F1 score is the harmonic mean of precision and sensitivity
        f1 = 2*(PPV*TPR)/(PPV+TPR) = (2*TP)/(2*TP+FP+FN)
    '''
    factors = factors.copy()
    licznik = 2 * factors.loc[:,'ppv'] * factors.loc[:,'tpr']
    #print(f'licznik:\n{licznik}\n\n')
    mian = factors.loc[:,['ppv','tpr']].sum(axis=1)
    return licznik/mian


#def mcc(factors):
    ''' Matthews correlation coefficient (MCC)
        mcc = (TP*TN - FP*FN) / [(TP+FP) * (TP+FN) * (TN+FP) * (TN+FN)]^0.5
    '''


def fm(factors):
    ''' Fowlkes–Mallows index (FM)
        fm = [(TP/(TP+FP))*(TP/(TP+FN))]^0.5 = (PPV * TPR)^0.5
    '''
    factors = factors.copy()
    licznik = (factors.loc[:,'ppv'] * factors.loc[:,'tpr'])**0.5
    mian = 1
    return licznik/mian


def bm(factors):
    ''' informedness or Fowlkes–Mallows index (BM)
        bm = TPR + TNR - 1
    '''
    factors = factors.copy()
    licznik = factors.loc[:,'tpr'] + factors.loc[:,'tnr'] - 1
    mian = 1
    return licznik/mian


def mk(factors):
    ''' markedness (MK) or deltaP
        mk = PPV + NPV - 1
    '''
    factors = factors.copy()
    licznik = factors.loc[:,'ppv'] + factors.loc[:,'npv'] - 1
    mian = 1
    return licznik/mian


# ================================================================================

# Funkcje do zapisywania i raportowania
# 






#########################################################################################################    
#########################################################################################################


if __name__ == '__main__':
    print('...')
    prl = printList().printL
    prd = printDict().printD

    # ================================================================================
    # 1. Parsowanie argumentów wejścia
    # ================================================================================
    args = parsujArgumenty()
    args = przygotujDaneInput(args)
    
    if args.verbose:
        print('1. Argumenty:')
        prd(vars(args))
    
    
    # ================================================================================
    # 2. Odczyt danych wejściowych
    # ================================================================================
    
    #_2.1. Dane po klasyfikacji
    if not args.cross:
        # dane wejściowe to dane po klasyfikacji - przynajmniej 3 kolumny
        data = pd.read_csv(args.input,sep=';')
        #data.astype({'predicted':np.uint8})
        idx = data.predicted.isna()
        print(f'isnan??:\n{data[idx]}\n\n')
    
        # 2.1.1. Pobierz dane referencyjne
        
        ref = pd.read_csv(args.ref,sep=';')
        
        if args.verbose:
            print(f'2.1.1. Dane referencyjne:\n{ref}\n')
        
        # 2.1.2. Macierz confusion
        cros = crossT(data,ref)
        #cros = walidujCross(cros,)
    
    
    
        # 
        
        

        # 2.1.1. Waliduj cros: mogą być różne liczby kolumn i wierszy - gdy nie ma wszystkich klas
        # wyrównaj liczbę wierszy i kolumn



    
    #_ 2.2. Cros matrix zamiast danych po klasyfikacji
    else:
        #dane wejściowe to cross matrix - tworzy pustą dataFrame jako 'data'
        data = pd.DataFrame()
        cros = pd.read_csv(args.input,sep=';',index_col=0)
        cros = cros.astype('int')
        
        if args.revers:
            # zmiana układu kolumn w macierzy na: kolumny to klasyfikacja, wiersze referencje
            cros = cros.T
        
    if args.verbose:
        # printuj dane wejściowe
        print('2. Dane wejściowe i confiusion matrix:')
        
        if not data.empty:
            print(f'\t2.1. Dane - wymiar danych: {data.shape}\n')
            if data.shape[1] > 10:
                print(data.iloc[:,:10],'\n')
            else:
                print(data,'\n')
        else:
            print('\t2.1. Dane wejściowe to cross matrix!\n')
        
        # printuj cross matrix
        print(f'\t2.2. Cross matrix:\n{cros}\n\n')



    # ================================================================================
    # 3. Utworzenie tabeli True/False Positive/Negative
    # ================================================================================    

    trueFalse = trueFalseTable(cros)
    if args.verbose:
        print(f'3. True/false table:\n{trueFalse}\n\n')    

    
    
    # ================================================================================
    # 4. Tradycyjne, klasyczne wskaźniki dokładności
    # ================================================================================
    
    oacc = overallAccuracy(cros)

    erom = errorsOfOmission(cros)

    ercom = errorsOfCommission(cros)

    producer = producerAcc(cros)

    user = userAcc(cros)
    
    oacc1 = [oacc for i in range(erom.shape[0])]
    
    classicAcc = pd.DataFrame({'Overall':oacc1, 'user':user, 'producer': producer, 'OmissionEr':erom, 'CommissionEr':ercom})

    if args.verbose:
        print(f'''4. Tradycyjne, klasyczne zestawienie dokładności:\n''')
        print(f'''\t4.1. Overall Accuracy: {oacc}\n''')
        print(f'''\t4.2. Zestawienie dokładności:\n{classicAcc}\n''')
    


    # ================================================================================
    # 5. Nowoczesne wskaźniki dokładności
    # ================================================================================

    # 5.1. Wskaźniki podstawowe 'factors1'
    
    modern1 = pd.DataFrame([],columns = list(factors1.keys()))
    
    for key, val in factors1.items():
        # przypisz do 'ff' odpowienią funkcję liczącą dany wskaźnik
        ff = eval(key)
        modern1.loc[:,key] = ff(trueFalse)

    modern1 = modern1.apply(np.round,decimals=7)
    
    if args.verbose:
        tmp = modern1*100
        tmp = tmp.apply(np.round,decimals=0)
        print(f'''5. Nowoczesne zestawienie dokładności:\n''')
        print(f'''\t5.1. Wskaźniki podstawowe 'factors1':\n{tmp}\n''')

    

    
    # 5.2. Wskaźniki dodatkowe 'factors2' na podstawie tabeli 'modern1'
    
    modern2 = pd.DataFrame([],columns = list(factors2.keys()))
    
    for key, val in factors2.items():
        # przypisz do 'ff' odpowienią funkcję liczącą dany wskaźnik
        ff = eval(key)
        modern2.loc[:,key] = ff(modern1)

    modern2 = modern2.apply(np.round,decimals=7)

    if args.verbose:
        tmp = modern2*100
        tmp = tmp.apply(np.round,decimals=0)
        print(f'''\t5.2. Wskaźniki dodatkowe 'factors2':\n{tmp}\n''')    
    
    

    # ================================================================================
    # 6. Zapisywanie danych
    # ================================================================================

    if args.save:
        if args.verbose:
            print(f'''\t6.1. Polecenia zapisywania danych:\n''')
            
        for key,val in args.save.items():
            polecenie = f'''{key}.to_csv('{val}',sep=';')'''
            if args.verbose:
                print(f'\t{key}:   {polecenie}')
            eval(polecenie)
            
    
    dataFrames ={}
    if args.raport and args.save:
        if args.verbose:
            print(f'''\n\n\t6.2. Polecenia tworzenia raportu:\n''')
            
        tmpLoader = jinja2.FileSystemLoader(args.runDir)
        env = jinja2.Environment(loader=tmpLoader)
        # template
        try:
            tm = env.get_template('raportForm.html')
        except jinja2.exceptions.TemplateNotFound:
            print('''\n\n\tRaport nie został utworzony!!!
                
                Formularz raportu, plik 'raportForm.html' nie został znaleziony w katalogu
                z którego został uruchomiony skrypt!!!\n\n''')
        
        else:
            for key,val in args.save.items():
                polecenie = f'''{key}.to_html()'''
                if args.verbose:
                    print(f'\t{key}:   {polecenie}')
                dataFrames[key] = eval(polecenie)
            
            with open(args.raport,'w') as f:
                f.write(tm.render(dataFrames=dataFrames))
    
    else:
        print(f'''\nNie podano opcji '-s' (save) - raport nie zostanie wykonany!\n\n''')
        
    print('\n\n......Skrytp zakończony pozytywnie:\n\n')
