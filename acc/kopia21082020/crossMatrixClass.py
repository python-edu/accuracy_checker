# -*- coding: utf-8 -*-

wersja = 'w1.2020.06.30'
opis = '''
Wersja skryptu: {0}.

Skrypt tworzy 'cross matrix' z danych zawierających etykiety prawdziwe i predicted.

'''.format(wersja)


import sys
import pandas as pd

sys.path.append('/home/u1/03_Programowanie/03Python/skrypty/skryptyCht2@agh/generals/')

from printing import printDict, printList
import numpy as np

#########################################################################################################

#-----------------------------------------------------------------------------------------------------------------------
class ConfusionMatrix():
    def __init__(self,data, trueValues):
        self.data = data.copy()
        self.trv = self._trueValues(trueValues)
        
        # utwórz cross tab w podstawowej wersji
        self.cros = self._crossTab()
        # wyrównaj liczbę wierszy i kolumn
        self.cros = self._wyrownajCros()
        
        # usuń kolumnę/wiersz 'fillVal' jeśli są w nich same zera
        # self._noFillVal: 0 lub 1 - 1 wskazuje, że usunięte zostały kolumna i wiersz 'fillVal'
        # usuń z trueValues 'fillVal' - jeśli usunięto z cross 'fillVal'
        self.cros, self._noFillVal = self._checkFillVallColumn()
        self.trv = self._usnFillValzTrv()
        
        #_sortujCros
        self.cros = self._sortujCros()
    
        # zamień liczby na nazwy
        self.cros = self._etykietyLiczboweNaStringi()
        # dodaj sumy do cross
        self.cros = self._podsumowanie()
        
        # utwórz wersję cros z opisami słownymi - crosFull
        self.crosFull = self._crosFull()
        
    #------------------------------------------------------------------
    
    
    def _trueValues(self,trueValues):
        trv = trueValues.copy()
        # pomiń wiersz z noData
        trv = trv[trv.loc[:,'short'] != 'noData']
        return trv.sort_values('true')        


    def _crossTab(self):
        return pd.crosstab(self.data.true, self.data.predicted, rownames=['true'],colnames=['predicted'])

    def _wyrownajCros(self):
        ''' Jeśli wyniki klasyfikacji nie zawierają wszystkich prawdziwych klas (jakiejś klasy nie wykryto),
            to cross matrix ma mniej kolumn(klasyfikacja) niż wierszy prawdy. Trzeba dodać kolumny z brakującymi
            klasami z zerowymi wystąpieniami. Podobnie jest z wierszami.
            
            Jeśli nie występują brakujące wartości to kolumna/wiersz 'fillVal' są zbędne (same zera) - usuń je.
        '''
        cros = self.cros.copy()
        s1 = set(self.trv.true.values) # wartości prawdziwe z pliku referencjego
        s2 = set(cros.columns.values) # kolumny cross
        s3 = set(cros.index.values) # wiersze cross

        # uzupełnia kolumny
        if len(s1.difference(s2)) > 0:
            dif = s1.difference(s2)
            for n in dif:
                cros.loc[:,n] = [0 for i in range(cros.shape[0])]
     
        # uzupełnia wiersze
        if len(s1.difference(s3)) > 0:
            dif = s1.difference(s3)
            for n in dif:
                cros.loc[n,:] = [0 for i in range(cros.shape[1])]

        return cros


    def _sortujCros(self):
        cros = self.cros.copy()
        kols = list(cros.columns.values)
        kols.sort()
        cros = cros.loc[:,kols]

        rows = list(cros.index.values)
        rows.sort()
        cros = cros.loc[rows,:]
        return cros

        
    def _checkFillVallColumn(self):
        '''' Jeśli nie występują brakujące wartości to kolumna/wiersz 'fillVal' są zbędne (same zera) - usuń je
             pobierz id 'fillVal' - tak nazywa się wiersz i kolumna z fillVal'''
        cros = self.cros.copy()
        trv = self.trv.copy()
        fillVal = trv.true[trv.short == 'fillVal'].iat[0]
    
        if cros.loc[fillVal,:].sum() == 0 and cros.loc[:,fillVal].sum() == 0:
            cros.drop(fillVal,axis=0,inplace=True)
            cros.drop(fillVal,axis=1,inplace=True)
            noFillVal = 1 # oznacza, że usunięto kolumny i wiersze 'fillVal'
        else:
            noFillVal = 0
    
        return cros, noFillVal


    def _usnFillValzTrv(self):
        ''' Jeśli nie ma w cros kolumny i wiersza 'fillVal' to
            usuwa te wartości z danych prawdziwych 'trv - trueValues'
        '''
        trv = self.trv.copy()
        if self._noFillVal:
            idx = trv.index[trv.short == 'fillVal'][0]
            trv.drop(idx,axis=0,inplace=True)
        return trv


    def _etykietyLiczboweNaStringi(self):
        cros = self.cros.copy()
        kols = [str(x) for x in cros.columns]
        rows = [str(x) for x in cros.index]
        cros.columns = kols
        cros.index = rows
        return cros

    
    def _podsumowanie(self):
        cros = self.cros.copy()
        sumRow = cros.sum(axis=1).to_numpy()
        cros.loc[:,'sumRow'] = sumRow
        
        sumKol = cros.sum(axis=0).to_numpy()
        cros.loc['sumCol',:] = sumKol
        
        return cros.astype('int')
    
    
    def _crosFull(self):
        ''' Tworzy nową cross matrix w której zmienia nazwy kolumn i wierszy
            z liczb na opisy słowne np. na owiec, zyto itp.
        '''
        crosFull = self.cros.copy()
        trv = self.trv.copy()
        
        #pobierz aktualne liczbowe nazwy kolumn i wierszy
        old = crosFull.columns.to_list()
        #print(f'\ntuuu old:\n{old}\n\n')
        old = [int(x) for x in old[:-1]]
        kols = []
        
        for x in old:
            idx = trv.loc[:,'true'] == x
            nazwa = trv.loc[idx,'short'].iat[0]
            kols.append(nazwa)
        #kols = trv.loc[:,'short'].to_list()
        rows = kols[:]
        
        kols.append('sumRow')
        rows.append('sumCol')
        
        crosFull.columns = kols
        crosFull.index = rows
        crosFull.axes[1].name='predicted'
        crosFull.axes[0].name='true'
        
        return crosFull

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
    print('''\nTesty klasy 'ConfusionMatrix()'!\n''')
    prd = printDict().printD
    # Odczyt danych testowych:
    # true values
    trv = pd.DataFrame({'true' :[1,2,3,9,10], 'short':['woda','las','trawa','fillVal','noData']})
    
    # dane testowe i odpowiedź
    data = pd.DataFrame({\
        'nazwa':['woda','woda','woda','woda','las','las','las','las','las','trawa','trawa','trawa','trawa'],\
         'true':[  1,    1,     1,     1,     2,    2,    2,    2,    2,    3,      3,      3,       3],    \
    'predicted':[  9,    1,     3,     1,     2,    1,    2,    3,    2,    2,      2,      3,       3]})

    wyn = pd.DataFrame({\
        'woda':[2,0,1,1,4], 'las':[1,3,1,0,5], 'trawa':[0,2,2,0,4], 'fillVal':[0,0,0,0,0], 'sumCol':[3,5,4,1,13]},\
        index = ['woda','las','trawa','fillVal','sumRow']).T
    
    # dane testowe nr 2 i odpowiedź - bez fillVal czyli wartości 9

    data1 = pd.DataFrame({\
        'nazwa':['woda','woda','woda','woda','las','las','las','las','las','trawa','trawa','trawa','trawa'],\
         'true':[  1,    1,     1,     1,     2,    2,    2,    2,    2,    3,      3,      3,       3],    \
    'predicted':[  1,    1,     3,     1,     2,    1,    2,    3,    2,    2,      2,      3,       3]})


    wyn1 = pd.DataFrame({\
        'woda':[3,0,1,4], 'las':[1,3,1,5], 'trawa':[0,2,2,4], 'sumCol':[4,5,4,13]},\
        index = ['woda','las','trawa','sumRow']).T
    

    # inicjacja instancji - data
    crs = ConfusionMatrix(data,trv)
    print(f'data:\n{data}\n\ncros:\n{crs.crosFull}\n\npoprawny:\n{wyn}\n')
    assert (wyn.equals(crs.crosFull)), 'Błąd'

    # inicjacja instancji - data1    
    crs = ConfusionMatrix(data1,trv)
    assert (wyn1.equals(crs.crosFull)), 'Błąd'
    
    print(f'''\nTesty klasy 'ConfusionMatrix()' zakończone pomyślnie!\n''')
