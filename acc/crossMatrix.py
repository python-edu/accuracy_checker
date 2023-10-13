# -*- coding: utf-8 -*-

wersja = 'w2.2020.11.30'
opis = '''
Wersja skryptu: {0}.

Skrypt tworzy 'cross matrix' z danych zawierających etykiety prawdziwe i predicted.

'''.format(wersja)


import sys
import pandas as pd

#sys.path.append('/home/u1/03_Programowanie/03Python/skrypty/skryptyCht2@agh/generals/')
sys.path.append('/home/u1/03_Programowanie/03Python/00Moduly/')


import os, json, jinja2
import argparse    # moduł zalecany w pythonie do parsowania argumentów lini poleceń
import textwrap
from prprint import printDict, printList
from indeksy import accIndex

from pathlib import Path

import numpy as np
from tabulate import tabulate



#########################################################################################################

def parsujArgumenty():
    '''
    '''
    
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter,description=opis)
    
    
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
    

    parser.add_argument('-s','--save',   help=textwrap.fill('''Powoduje, że wyniki zapisane zostaną do osobnych plików
                                                            csv: cros.csv, trueFalse.csv, classicAcc.csv, modern1.csv,
                                                            modern2.csv.''', width = 100), action = 'store_true')

    parser.add_argument('-rap','--raport',   help=textwrap.fill('''Generuje raport w html, czyli wszytskie tabele
                                                            w jednym pliku html - raport.html''', width = 100), action = 'store_true')
    
    parser.add_argument('-rf','--ref',   help=textwrap.fill('''Adres pliku 'csv' z danymi referencyjnymi - 3 kolumny:
                                                            'true;short;long'. Domyślnie adres pliku 'input' z dodatkowym członem 'true' np.:
                                                            - input: 'ndviKlasyfik.csv'
                                                            - ref:  'ndviKlasyfik_true.csv'.''', width = 100), default = 'default')
    
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
        save = ['cros', 'crosFull']
        sl ={}
        tmpName = Path(args.input).name.split('.')[0]
        for i,s in enumerate(save,1):
            name = f'res_{tmpName}_{s}.csv'
            sl[s] = Path(args.input).with_name(name).resolve().as_posix()
        args.save = sl
   
    if args.raport:
        name = Path(args.input).name.split('.')[0]
        name = f'{name}_cros.html'
        args.raport = Path(args.input).with_name(name).as_posix()
     
    if args.ref == 'default':
        name = Path(args.input).name.split('.')[0]
        name = f'{name}_true.csv'
        args.ref = Path(args.input).with_name(name).as_posix()
    else:
        args.ref = Path(args.ref).resolve().as_posix()
    
    return args




#-----------------------------------------------------------------------------------------------------------------------

def crossT(data,ref):
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
        
    # wersja cross matrix bez nazw kolumn i wierszy tylko z etykietami liczbowymi
    crosFull = cros.copy()
    
    # zamień liczby na liczby stringi w nazwach kolumn i wierszy
    kols = [str(x) for x in cros.columns]
    rows = [str(x) for x in cros.index]
    cros.columns = kols
    cros.index = rows
    
    cros.axes[1].name='predicted'
    cros.axes[0].name='true'
    
    crosFull.columns = ref.loc[:,'short']
    crosFull.index = ref.loc[:,'short']
    crosFull.axes[1].name='predicted'
    crosFull.axes[0].name='true'
    
    # dodaje sumy wierszy i kolumn
    sum1 = crosFull.sum(axis=0)
    crosFull.loc['sumCol',:] = sum1
    cros.loc['sumCol',:] = sum1.to_list()
    
    sum2 = crosFull.sum(axis=1)
    crosFull.loc[:,'sumRow'] = sum2
    cros.loc[:,'sumRow'] = sum2.to_list()
    
    return cros.astype('int'), crosFull.astype('int')




#-----------------------------------------------------------------------------------------------------------------------

def crossT1(data,ref):
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
    print(f'\nCross tuu!:\n{cros}\n\n')
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
    #print(f'\ntrue: {s1}\nref:\n{s2}\n{s3}\n')
    
    # uzupełnia kolumny
    if len(s1.difference(s2)) > 0:
        dif = s1.difference(s2)
        for n in dif:
            cros.loc[:,n] = [0 for i in range(cros.shape[0])]
            
        kols = list(cros.columns.values)
        kols.sort()
        cros = cros.loc[:,kols]
     
     # uzupełnia wiersze
    if len(s1.difference(s3)) > 0:
        dif = s1.difference(s3)
        for n in dif:
            cros.loc[n,:] = [0 for i in range(cros.shape[1])]
            
        rows = list(cros.index.values)
        rows.sort()
        cros = cros.loc[rows,:]
    
    # Jeśli nie występują brakujące wartości to kolumna/wiersz 'fillVal' są zbędne (same zera) - usuń je
    # pobierz id 'fillVal' - tak nazywa się wiersz i kolumna z fillVal
    fillVal = ref.true[ref.short == 'fillVal'].iat[0]
    
    if cros.loc[fillVal,:].sum() == 0 and cros.loc[:,fillVal].sum() == 0:
        cros.drop(fillVal,axis=0,inplace=True)
        cros.drop(fillVal,axis=1,inplace=True)
        noFillVal = 1 # oznacza, że usunięto kolumny i wiersze 'fillVal'
    else:
        noFillVal = 0
    
    return cros, noFillVal





#-----------------------------------------------------------------------------------------------------------------------

def walidujCros1(cros,ref):
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

    # dane wejściowe to dane po klasyfikacji - przynajmniej 3 kolumny
    data = pd.read_csv(args.input,sep=';')
    #data.astype({'predicted':np.uint8})
    idx = data.predicted.isna()
    print(f'isnan??:\n{data[idx]}\n\n')
    
    # 2.1.1. Pobierz dane referencyjne
        
    ref = pd.read_csv(args.ref,sep=';')
        
    if args.verbose:
        print(f'2.1.1. Dane referencyjne:\n{ref}\n')
        
    # 2.1.2. Macierz confusion - dwie wersje!!! z nazwami kolumn/wierszy i bez nazw (liczby zamiast tego)
    cros, crosFull = crossT(data,ref)   
        
    if args.verbose:
        # printuj dane wejściowe
        print('2. Dane wejściowe i confiusion matrix:')
        print(f'\t2.1. Dane - wymiar danych: {data.shape}\n')
        if data.shape[1] > 10:
            print(data.iloc[:,:10],'\n')
        else:
            print(data,'\n')
 
        # printuj cross matrix
        print(f'\t2.2. Cross matrix:\n{cros}\n\n')



    # ================================================================================
    # 3. Zapisywanie danych
    # ================================================================================

    if args.save:
        if args.verbose:
            print(f'''\t6.1. Polecenia zapisywania danych:\n''')
            
        for key,val in args.save.items():
            polecenie = f'''{key}.to_csv('{val}',sep=';')'''
            if args.verbose:
                print(f'\t{key}:   {polecenie}')
            eval(polecenie)
    
    # raport
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
