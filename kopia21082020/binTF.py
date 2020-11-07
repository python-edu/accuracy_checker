# -*- coding: utf-8 -*-

wersja = 'w1.2020.06.30'
opis = '''
Wersja skryptu: {0}.

Skrypt tworzy table wskaźników TF, TN, FP, FN zwaną 'binTF' na podstawie
'cross matrix.

'''.format(wersja)


import sys
import pandas as pd

sys.path.append('/home/u1/03_Programowanie/03Python/skrypty/skryptyCht2@agh/generals/')

from printing import printDict, printList
import numpy as np

#########################################################################################################

#-----------------------------------------------------------------------------------------------------------------------
class BinTFtable():
    def __init__(self,data, shape='h'):
        '''
            Args:
                - data:     pd.DataFrame, ConfusionMatrix
                - shape:    str, h-horyontalnie, v-vertykalnie, określa układ tabeli wynikowej
        '''
        self.cros, self._crosNames = self._cros(data)
        self.shape = shape
        self._rowNames = ['TP','TN','FP','FN']
        

        # utwórz wersję cros z opisami słownymi - crosFull
        self.binTF = self._binTable()
        if self.shape == 'v':
            self.binTF = self.binTF.T
    
    
    # -----------------------------------------------------------------
    
    def _cros(self,cros):
        cros = cros.copy().iloc[:-1,:-1]
        return cros.to_numpy(), cros.columns.to_list()
    # -----------------------------------------------------------------

    def _binTable(self):
        ar = self.cros.copy()
        cols = self._crosNames # nazwy kolumn z cros
        sl = {}
        #rowsIdx = ['TP','TN','FP','FN']
    
        for i in range(ar.shape[0]):
            tp = ar[i,i]
        
            tmp = np.delete(ar,i,1)  # usuń kolumnę rozpatrywanej klasy
            tmp = np.delete(tmp,i,0) # usuń wiersz rozpatrywanej klasy

            tn = tmp.sum()
        
            row = np.delete(ar[i,:],i) # pobierz wiersz i usuń z niego rozpatrywaną kolumnę
            fn = row.sum()
        
            col = np.delete(ar[:,i],i) # pobierz kolumnę i usuń z niej rozpatrywany wiersz
            fp = col.sum()
        
            sl[cols[i]] = [tp,tn,fp,fn]
    
        wyn = pd.DataFrame(sl,index=self._rowNames)
        #wyn.loc['total',:] = wyn.sum(axis=0)
        return wyn.astype('int')


#########################################################################################################    
#########################################################################################################


if __name__ == '__main__':
    print('''\nTesty klasy 'BinTFtable'!\n''')
    prd = printDict().printD
    
    # dane testowe i wyniki:
    nazwy = ['water','forest','urban'] # nazwy kolumn i wierszy (są takie same)
    value = [[21,5,7],[6,31,2],[0,1,22]] # liczby w komórkach
    cros1 = pd.DataFrame(value,columns=nazwy,index=nazwy)
    total = cros1.sum(axis=0)
    cros1.loc['total',:] = total
    total = cros1.sum(axis=1)
    cros1.loc[:,'total'] = total
    cros1 = cros1.astype('int')

    #wyniki
    w1 = pd.DataFrame([[21., 31., 22.],[56., 50., 63.],[ 6.,  6.,  9.],[12.,  8.,  1.]],columns=nazwy,\
                        index=['TP','TN','FP','FN']).astype('int')
    
    #print(f'\ncros1:\n{cros1}\n\nwynik1:\n{w1}\n')
    
    nazwy = [1,2,3] # nazwy kolumn i wierszy (są takie same)
    value = [[21,5,7],[6,31,2],[0,1,22]] # liczby w komórkach
    cros2 = pd.DataFrame(value,columns=nazwy,index=nazwy)
    total = cros2.sum(axis=0)
    cros2.loc['total',:] = total
    total = cros2.sum(axis=1)
    cros2.loc[:,'total'] = total
    cros2 = cros2.astype('int')

    #wyniki
    w2 = pd.DataFrame([[21., 31., 22.],[56., 50., 63.],[ 6.,  6.,  9.],[12.,  8.,  1.]],columns=nazwy,\
                        index=['TP','TN','FP','FN']).astype('int')
    #print(f'\ncros2:\n{cros2}\n\nwynik2:\n{w2}\n')

    tf = BinTFtable(cros1)
    #print(f'\ntf.binTF:\n{tf.binTF}\n')
    
    assert (tf.binTF.equals(w1)), 'Błąd'
    
    tf = BinTFtable(cros2)
    #print(f'\ntf.binTF:\n{tf.binTF}\n')    
    #assert (tf.binTF.equals(w2)), 'Błąd'

    print(f'.....Testy zakończone pozytywnie!\n')
