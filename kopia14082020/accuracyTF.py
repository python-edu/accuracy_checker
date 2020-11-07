# -*- coding: utf-8 -*-

wersja = 'w1.2020.06.30'
opis = '''
Wersja skryptu: {0}.

Skrypt sprawdza jakość klasyfikacji:
   - wykorzystuje table prawdy i fałszu 'true/false'
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
    
    
    parser.add_argument('input',  type=str,   help=textwrap.fill(f'''Adres pliku 'csv' z danymi. Dane to:
                                                                1. True/False: tabela o 4 wieszach i kolumnach odpowiadających klasom, np:
                                                                ----------------------------    
                                                                |    |owies| zyto| ... |
                                                                | ---| --- | --- |---- |
                                                                | TP |  1  |  55 | ... |
                                                                | TN | 15  |  99 | ... |
                                                                | FP |  5  |   3 | ... |
                                                                | FN | 33  |  46 | ... |
                                                                ----------------------------
                                                                :
                                                                ''',width = 70))
    

    parser.add_argument('-s','--save',   help=textwrap.fill('''Powoduje, że wyniki zapisane zostaną do osobnych plików
                                                            csv: classicAcc.csv, modern1.csv, modern2.csv.''', width = 100), action = 'store_true')

    parser.add_argument('-rap','--raport',   help=textwrap.fill('''Generuje raport w html, czyli wszytskie tabele
                                                            w jednym pliku html - raport.html''', width = 100), action = 'store_true')
    
    parser.add_argument('-sp','--sep', type=str,  help=textwrap.fill('''Określa separator kolumn pliku csv. Domyślnie to średnik ';'.''', width = 100), default=';')

    parser.add_argument('-p','--prec', type=int,  help=textwrap.fill('''Z jaką precyzją zapisać wyniki obliczeń do pliku. Domyślnie 6 po przecinku.''', width = 100), default = 6)

    parser.add_argument('-v','--verbose',   help=textwrap.fill(u"Wyświetla bardziej szczegółowe informacje.", width = 100), action = 'store_true')
    
    args = parser.parse_args() 
    return args


#-----------------------------------------------------------------------------------------------------------------------

def przygotujDaneInput(args):
    '''Funkcja przetwarza argumenty wejściowe skryptu'''
    
    # ustal katalog z którego uruchamiany jest skrypt
    # na potrzeby przygotowania raportu - tam jest szablon raportu!
    args.runDir = Path(__file__).resolve().parent.as_posix()
    
    if args.input:
        args.input = Path(args.input).resolve()
        #args.out = args.input.with_name('cross.csv').resolve().as_posix()
        #args.workDir = args.input.parent.as_posix()
        args.input = args.input.as_posix()
    
    # tworzy adresy do zapisu wyników
    if args.save:
        save = ['trueFalse','classicAcc', 'modern1','modern2']
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



#-----------------------------------------------------------------------------------------------------------------------


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

def overallAccuracy(trueFalse,prec=2):
    tf = trueFalse.copy().astype(np.float128)
    sumaTp = tf.loc['TP'].sum()
    sumaAll = tf.sum(axis=0).iat[0]
    #return np.round((sumaTp/sumaAll)*100,prec)
    return np.round((sumaTp/sumaAll),prec)

def errorsOfOmission(trueFalse,prec=2):
    tf = trueFalse.copy().astype(np.float128)
    sumR = tf.loc[['TP','FN'],:].sum(axis=0)
    erOm = tf.loc['FN',:] / sumR
    return np.round(erOm,prec)

def errorsOfCommission(trueFalse,prec=2):
    tf = trueFalse.copy().astype(np.float128)
    sumK = tf.loc[['TP','FP'],:].sum(axis=0)
    erCom = tf.loc['FP',:] / sumK
    return np.round(erCom,prec)

def producerAcc(trueFalse,prec=2):
    tf = trueFalse.copy().astype(np.float128)
    sumR = tf.loc[['TP','FN'],:].sum(axis=0)
    producer = tf.loc['TP',:] / sumR
    return np.round(producer,prec)

def userAcc(trueFalse,prec=2):
    tf = trueFalse.copy().astype(np.float128)
    sumK = tf.loc[['TP','FP'],:].sum(axis=0)
    user = tf.loc['TP',:] / sumK   
    return np.round(user,prec)

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

def acc(trueFalse,prec=2):
    ''' accuracy (ACC):
        ACC = (TP+TN)/(P+N) = (TP+TN)/(TP+TN+FP+FN)
    '''
    tf = trueFalse.copy().astype(np.float128)
    licznik = tf.loc[['TP','TN'],:].sum(axis=0)
    mian = tf.loc[['TP','TN','FP','FN'],:].sum(axis=0)
    #print(f'\n\nSprrrrrr:\n\n{licznik}\n\n{mian}\n\nkonirc\n\n')
    return licznik/mian


def ppv(trueFalse,prec=2):
    ''' precision or positive predictive value (PPV)
        PPV = TP / (TP + FP)
    '''
    tf = trueFalse.copy().astype(np.float128)
    licznik = tf.loc['TP',:]
    mian = tf.loc[['TP','FP'],:].sum(axis=0)
    return licznik/mian


def tpr(trueFalse,prec=2):
    ''' sensitivity, recall, hit rate, or true positive rate (TPR)
        TPR = TP/P = TP/(TP + FN) = 1 − FNR '''
    tf = trueFalse.copy().astype(np.float128)
    licznik = tf.loc['TP',:]
    mian = tf.loc[['TP','FN'],:].sum(axis=0)
    return licznik/mian


def tnr(trueFalse,prec=2):
    ''' specificity, selectivity or true negative rate (TNR)
        TNR = TN/N = TN/(TN + FP) = 1 − FPR'''
    tf = trueFalse.copy().astype(np.float128)
    licznik = tf.loc['TN',:]
    mian = tf.loc[['TN','FP'],:].sum(axis=0)
    return licznik/mian


def npv(trueFalse,prec=2):
    ''' negative predictive value (NPV)
        NPV = TN/(TN + FN) = 1 − FOR'''
    tf = trueFalse.copy().astype(np.float128)
    licznik = tf.loc['TN',:]
    mian = tf.loc[['TN','FN'],:].sum(axis=0)
    return licznik/mian


def fnr(trueFalse,prec=2):
    ''' miss rate or false negative rate (FNR)
        FNR = FN/P = FN/(FN + TP) = 1 − TPR'''
    tf = trueFalse.copy().astype(np.float128)
    licznik = tf.loc['FN',:]
    mian = tf.loc[['FN','TP'],:].sum(axis=0)
    return licznik/mian


def fpr(trueFalse,prec=2):
    ''' fall-out or false positive rate (FPR)
        FPR = FP/N = FP/(FP + TN) = 1 − TNR'''
    tf = trueFalse.copy().astype(np.float128)
    licznik = tf.loc['FP',:]
    mian = tf.loc[['FP','TN'],:].sum(axis=0)  
    return licznik/mian


def fdr(trueFalse,prec=2):
    ''' false discovery rate (FDR)
        FDR = FP/(FP + TP) = 1 − PPV '''
    tf = trueFalse.copy().astype(np.float128)
    licznik = tf.loc['FP',:]
    mian = tf.loc[['FP','TP'],:].sum(axis=0)  
    return licznik/mian


def foRate(trueFalse,prec=2):
    ''' false omission rate (FOR)
        FOR = FN/(FN + TN) = 1 − NPV '''
    tf = trueFalse.copy().astype(np.float128)
    licznik = tf.loc['FN',:]
    mian = tf.loc[['FN','TN'],:].sum(axis=0)
    return licznik/mian


def ts(trueFalse,prec=2):
    ''' Threat score (TS) or critical success index (CSI)
        TS = TP/(TP + FN + FP) '''
    tf = trueFalse.copy().astype(np.float128)
    licznik = tf.loc['TP',:]
    mian = tf.loc[['TP','FN','FP'],:].sum(axis=0)
    return licznik/mian


def mcc(trueFalse,prec=2):
    ''' Matthews correlation coefficient (MCC)
        mcc = (TP*TN - FP*FN) / [(TP+FP) * (TP+FN) * (TN+FP) * (TN+FN)]^0.5
    '''
    tf = trueFalse.copy().astype(np.float128)
    
    tp = tf.loc['TP',:]
    tn = tf.loc['TN',:]
    fp = tf.loc['FP',:]
    fn = tf.loc['FN',:]
    
    licznik = (tp * tn) - (fp * fn)
    mian = ((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))**0.5
    #mm = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    
    #print(f'\nlicznik:\n{licznik}\n\nmiano:\n{mian}\n\nmcc:\n{licznik/mian}\n\n')
    return licznik/mian


#
#
# 2. Wskażniki złożone - obliczane na podstawie wskaźników prostych
#   - 

factors2 = {'pt':'prevalence threshold', 'ba':'balanced accuracy', 'f1':'harmonic mean of precision and sensitivity',\
            'fm':'Fowlkes–Mallows index',\
            'bm':'Fowlkes–Mallows index', 'mk':'markedness or deltaP'}


def pt(factors,prec=2):
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


def ba(factors,prec=2):
    ''' Balanced accuracy (BA):
        ba = (TPR + TNR)/2
    '''
    factors = factors.copy()
    licznik = factors.loc[:,['tpr','tnr']].sum(axis=1)
    mian = 2
    return licznik/mian


def f1(factors,prec=2):
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


def fm(factors,prec=2):
    ''' Fowlkes–Mallows index (FM)
        fm = [(TP/(TP+FP))*(TP/(TP+FN))]^0.5 = (PPV * TPR)^0.5
    '''
    factors = factors.copy()
    licznik = (factors.loc[:,'ppv'] * factors.loc[:,'tpr'])**0.5
    mian = 1
    return licznik/mian


def bm(factors,prec=2):
    ''' informedness or Fowlkes–Mallows index (BM)
        bm = TPR + TNR - 1
    '''
    factors = factors.copy()
    licznik = factors.loc[:,'tpr'] + factors.loc[:,'tnr'] - 1
    mian = 1
    return licznik/mian


def mk(factors,prec=2):
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
    
    trueFalse = pd.read_csv(args.input, sep=args.sep, index_col=0).astype('int')
    #trueFalse = trueFalse.astype('int')
    if args.verbose:
            print(f'''2. Dane wejściowe 'true/false':\n{trueFalse}\n''')

    # ================================================================================
    # 3. Utworzenie tabeli True/False Positive/Negative
    # ================================================================================    

 

    
    
    # ================================================================================
    # 4. Tradycyjne, klasyczne wskaźniki dokładności
    # ================================================================================
    
    oacc = overallAccuracy(trueFalse,args.prec)

    erom = errorsOfOmission(trueFalse,args.prec)

    ercom = errorsOfCommission(trueFalse,args.prec)

    producer = producerAcc(trueFalse,args.prec)

    user = userAcc(trueFalse,args.prec)
    
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

    modern1 = modern1.apply(np.round,decimals=args.prec)
    
    if args.verbose:
        print(f'''5. Nowoczesne zestawienie dokładności:\n''')
        print(f'''\t5.1. Wskaźniki podstawowe 'factors1':\n{modern1}\n''')

    

    
    # 5.2. Wskaźniki dodatkowe 'factors2' na podstawie tabeli 'modern1'
    
    modern2 = pd.DataFrame([],columns = list(factors2.keys()))
    
    for key, val in factors2.items():
        # przypisz do 'ff' odpowienią funkcję liczącą dany wskaźnik
        ff = eval(key)
        modern2.loc[:,key] = ff(modern1)

    modern2 = modern2.apply(np.round,decimals=args.prec)

    if args.verbose:
        print(f'''\t5.2. Wskaźniki dodatkowe 'factors2':\n{modern2}\n''')    
    
    

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
