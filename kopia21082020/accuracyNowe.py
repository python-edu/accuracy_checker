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
sys.path.append('/home/u1/03_Programowanie/03Python/00Moduly/')

import os, json, jinja2
import argparse    # moduł zalecany w pythonie do parsowania argumentów lini poleceń
import textwrap
from printing import printDict, printList
from raportBase import SimpleRaport
from indeksy import accClasic, accIndex, accClasicBin
from crossMatrixClass import ConfusionMatrix
from binTF import BinTFtable

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
    
    parser.add_argument('-d','--dataType', type=str,  help=textwrap.fill('''Mówi czym są dane wejściowe. Możliwości:
                                                        - 'data' dane po klasyfikacji
                                                        - 'cros' cros matrix
                                                        - 'bin' binTF.
                                                        Domyślnie 'data'.''', width = 100), default = 'data')
    
    parser.add_argument('-p','--precision', type=int,  help=textwrap.fill('''Dokładność. Domyślnie 4 miejsca po przecinku.''', width = 100), default = 4)
    

    parser.add_argument('-r','--revers',   help=textwrap.fill('''Wskazuje, że układ cross matrix jest odwrócony
                                                            tzn. w kolumnach są dane referencyjne a w wierszach 
                                                            wyniki klasyfikacji (predict).''', width = 100), action = 'store_true')

    parser.add_argument('-sp','--sep', type=str,  help=textwrap.fill('''Określa separator kolumn pliku csv. Domyślnie to średnik ';'.''', width = 100), default=';')

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
    
    args.runDir = Path(__file__).resolve().parent.as_posix() # po co?
    
    if args.input:
        args.input = Path(args.input).resolve()
        #args.out = args.input.with_name('cross.csv').resolve().as_posix()
        args.workDir = args.input.parent.as_posix()
        args.input = args.input.as_posix()
    

   
    if args.raport:
        name = Path(args.input).name.split('.')[0]
        name = f'{name}_raport.html'
        args.raport = Path(args.input).with_name(name).as_posix()
     
    if args.ref == 'default':
        name = Path(args.input).name.split('.')[0]
        name = f'{name}_true.csv'
        args.ref = Path(args.input).with_name(name).as_posix()
    else:
        args.ref = Path(args.ref).resolve().as_posix()
    
    # jeśli dane input to 'binTF' to niepotrzebny jest plik 'ref'
    if args.dataType == 'bin':
        args.ref = None
    
    return args




#-----------------------------------------------------------------------------------------------------------------------
class AccRaport(SimpleRaport):
     
    def _konvertujDane(self):
        ''' Dane w tym skrypcie to pd.DataFrame. Przeciążenie orygonalnej
        metody polega na konwersji dataFrame do html-a.
        Input:
            - self.data:    lista danych, tu pd.DataFrame
        '''
        htmlData = [dane.to_html() for dane in self.data]
        dataToRender = dict(zip(self.opis,htmlData))
        return dataToRender




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
    
    if args.dataType == 'data':
        data = pd.read_csv(args.input,sep=args.sep)
        ref  = pd.read_csv(args.ref,sep=args.sep)
        
        cr = ConfusionMatrix(data,ref)
        cros = cr.cros
        crosFull = cr.crosFull
        
        bin = BinTFtable(cros)
        binTF = bin.binTF
        wykaz =['data','ref','cros','crosFull','binTF']
        toSave=['ref','cros','crosFull','binTF']
        
    elif args.dataType == 'cros':
        ref  = pd.read_csv(args.ref,sep=args.sep)
        cros = pd.read_csv(args.input,sep=args.sep,index_col=0)
        bin = BinTFtable(cros)
        binTF = bin.binTF
        wykaz =['ref','cros','binTF']
        toSave=['binTF']
        
    else:
        binTF = pd.read_csv(args.input,sep=args.sep,index_col=0)
        # sprawdź w jakim układzie jest data DataFrame
        #print(f'\ntuu:\n{binTF}\n\n')
        kols = set(binTF.columns.to_list())
        spr = set(['TP','TN','FP','FN'])
        if spr.issubset(kols):
            binTF = binTF.T
            print(f'\ntuu:\n{binTF}\n\n')
        
        wykaz =['binTF']
        
        # w układzie pionowym - na potrzeby raportu
        binTFv = binTF.T
        toSave=['binTFv']
        cros = None
        
    if args.verbose:
        print(f'2. Dane wejściowe:\n\n') 
        for it in wykaz:
            print(f'''{it}:\n{eval(it)}\n\n''')    
    
 
    # ================================================================================
    # 3. Tradycyjne, klasyczne wskaźniki dokładności
    # ================================================================================
    
    if cros is not None:
        acc1 = accClasic(cros.iloc[:-1,:-1],args.precision)
        
    else:
        acc1 = accClasicBin(binTF,args.precision)
    
    classicAcc = acc1.tabela
    if args.verbose:
        print(f'''3. Tradycyjne, klasyczne zestawienie dokładności:\n\n{classicAcc}\n''')

    toSave.append('classicAcc')
    # ================================================================================
    # 4. Nowe wskaźniki dokładności
    # ================================================================================
    acc2 = accIndex(binTF,precision=args.precision)
    modern1 = {}
    modern2 = {}
    
    for k,v in vars(acc2).items():
        if k in ['acc', 'ppv', 'tpr','tnr','npv','fnr','fpr','fdr','foRate','ts','mcc']:
            modern1[k] = v
        elif k in ['pt', 'ba', 'f1', 'fm', 'bm', 'mk']:
            modern2[k] = v
    
    modern1 = pd.DataFrame(modern1)
    modern2 = pd.DataFrame(modern2)
    
    toSave.extend(['modern1','modern2'])
    

    if args.verbose:
        print(f'''\t4. Wskaźniki dodatkowe:\n\nmodern1:\n{modern1}\n\nmodern2:\n{modern2}\n''')
    
    # ================================================================================
    # 5. Zapisywanie danych
    # ================================================================================

    if args.save:
        if args.verbose:
            print(f'''\t6.1. Polecenia zapisywania danych:\n''')
            
        #for key,val in args.save.items():
        for nazwa in toSave:
            if nazwa == 'binTF':
                binTF = binTF.T
            ad = Path(args.workDir).joinpath(f'res_{nazwa}.csv')
            polecenie = f'''{nazwa}.to_csv('{ad}',sep='{args.sep}')'''
            if args.verbose:
                print(f'\t{nazwa}:   {polecenie}')
            eval(polecenie)
            
    # tworzy raport html
    if args.raport and args.save:
        if args.verbose:
            print(f'''\n\n\t6.2. Polecenia tworzenia raportu:\n''')
            
        
        dane = [eval(nazwa) for nazwa in toSave]
        raport = AccRaport(data=dane,opis=toSave)
        raport.saveRaport(raportAdres=args.raport)
        
    else:
        print(f'''\nNie podano opcji '-s' (save) - raport nie zostanie wykonany!\n\n''')
        
    print('\n\n......Skrytp zakończony pozytywnie:\n\n')
