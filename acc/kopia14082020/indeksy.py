# -*- coding: utf-8 -*-

wersja = 'w1.2020.08.13'
opis = '''
Wersja skryptu: {0}.

Moduł zawiera klasy dostarczające metody do oblicznia różnych wskźników dokładności.

Klasy:
  ## 1. 'accClasic'
  ## 2. 'accIndex'

'''.format(wersja)


import numpy as np
import pandas as pd


#########################################################################################################
# ----------------------------------------------------------------------------------------------------------------------

class accClasic:
    ''' Oblicza w sposób tradycyjny, na podstawie 'cross matrix' wartości dokładności i błędów
        klasyfikacji:
        - overallAccuracy, producerAcc, userAcc, errorsOfOmission, errorsOfCommission
        
        Dane wejściowe:   pd.DataFRame lub np.array, cross matrix - bez podsumowań wierszy i kolumn!!!!
    '''
    
    def __init__(self,data,precision=7,revers=False):
        # liczba miejsc po przecinku dla wyników
        self.precision = precision

        # data: cross matrix - pd.DataFrame
        #   - revers: wskazuje, że jest odwrócony układ cross matrix:
        #               * kolumny to prawda
        #               * wiersze to predicted
        if revers:
            self.data =  self._reverseData(data)
        else:
            self.data = data.copy()
    
        # oblicz podstawowe wartości: sumy w wierszach, kolumnach, całkowitą sumę
        self._obliczSummy()
        
        # oblicz wskaźniki/indeksy dokładności
        self._obliczIndeksy()


    # ....................................................

    def _reverseData(self):
        return self.data.T
        
    # ....................................................
    
    def _obliczSummy(self):
        self._total = self.data.to_numpy().sum()
        self._rowsSum = self.data.sum(axis=1) # sumy w wierszach
        self._colsSum = self.data.sum(axis=0)
        self._diagonalne = np.diagonal(self.data) # wartości diagonalne
        

    # ....................................................

    def _obliczIndeksy(self):
        self.accOver = self._overallAccuracy()
        self.accProd = self._producerAcc()
        self.accUser = self._userAcc()
        
        self.erOm = self._errorsOfOmission()
        self.erCom = self._errorsOfCommission()
        
    # ....................................................
    
    def _overallAccuracy(self):        
        allGood = self._diagonalne.sum()
        return np.round(allGood/self._total, self.precision)

    def _errorsOfOmission(self):        
        diff = self._rowsSum - self._diagonalne
        return np.round(diff / self._rowsSum, self.precision)

    def _errorsOfCommission(self):
        diff = self._colsSum - self._diagonalne
        return np.round(diff / self._colsSum, self.precision)

    def _producerAcc(self):
        return np.round(self._diagonalne / self._rowsSum, self.precision)

    def _userAcc(self):  
        return np.round(self._diagonalne / self._colsSum, self.precision)


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


class accIndex:
    ''' Oblicza szereg wskaźników dokładności stosowanych w ocenie klasyfikacji
        szczególnie w 'machine learning'.
        
        Dane wejściowe:   pd.DataFRame, tabela true/false w układzie:
                          ----------------------------    
                          |    |owies| zyto| ... |
                          | ---| --- | --- |---- |
                          | TP |  1  |  55 | ... |
                          | TN | 15  |  99 | ... |
                          | FP |  5  |   3 | ... |
                          | FN | 33  |  46 | ... |
                          ----------------------------
        
    '''
    
    def __init__(self,data,precision=7):
        self.tf = data.copy().astype(np.float128)
        self.precision = precision
        self.overMethodsM1()
        self.overMethodsM2()
        #for k,v in sorted(vars(self).items()):
         #   print(f'\n{k}:\n{v}\n')
        #print('\nvars(self):\n',vars(self))

    # ....................................................
    
    def overMethodsM1(self):
        ''' Wykonuje metody obliczające indeksy na podstawie wartości TP,TN,FP,FN z tabeli
            'binTF'. Wskaźniki te wcześniej były w grupie 'modern1'.
            '''
        for m in dir(accIndex):
            #if re.search(r'^_{1}[a-z]+',m):
            if callable(getattr(accIndex, m)) and m.startswith("_x1"):
                kod = f'''self.{m[3:]} = np.round(self.{m}(), self.precision)'''
                #print(f'{m}\t{kod}')
                exec(kod)
        
    def overMethodsM2(self):
        for m in dir(accIndex):
            if callable(getattr(accIndex, m)) and m.startswith("_x2"):
                kod = f'''self.{m[3:]} = np.round(self.{m}(), self.precision)'''
                #print(f'{m}\t{kod}')
                exec(kod)    


    # ....................................................

    def _x1acc(self):
        ''' accuracy (ACC):
            ACC = (TP+TN)/(P+N) = (TP+TN)/(TP+TN+FP+FN)
        '''
        licznik = self.tf.loc[['TP','TN'],:].sum(axis=0)
        mian = self.tf.loc[['TP','TN','FP','FN'],:].sum(axis=0)
        #print(f'\n\nSprrrrrr:\n\n{licznik}\n\n{mian}\n\nkonirc\n\n')
        return licznik/mian


    def _x1ppv(self):
        ''' precision or positive predictive value (PPV)
            PPV = TP / (TP + FP)
        '''
        licznik = self.tf.loc['TP',:]
        mian = self.tf.loc[['TP','FP'],:].sum(axis=0)
        return licznik/mian


    def _x1tpr(self):
        ''' sensitivity, recall, hit rate, or true positive rate (TPR)
            TPR = TP/P = TP/(TP + FN) = 1 − FNR '''
        licznik = self.tf.loc['TP',:]
        mian = self.tf.loc[['TP','FN'],:].sum(axis=0)
        return licznik/mian


    def _x1tnr(self):
        ''' specificity, selectivity or true negative rate (TNR)
            TNR = TN/N = TN/(TN + FP) = 1 − FPR'''
        licznik = self.tf.loc['TN',:]
        mian = self.tf.loc[['TN','FP'],:].sum(axis=0)
        return licznik/mian


    def _x1npv(self):
        ''' negative predictive value (NPV)
            NPV = TN/(TN + FN) = 1 − FOR'''
        licznik = self.tf.loc['TN',:]
        mian = self.tf.loc[['TN','FN'],:].sum(axis=0)
        return licznik/mian


    def _x1fnr(self):
        ''' miss rate or false negative rate (FNR)
            FNR = FN/P = FN/(FN + TP) = 1 − TPR'''
        licznik = self.tf.loc['FN',:]
        mian = self.tf.loc[['FN','TP'],:].sum(axis=0)
        return licznik/mian


    def _x1fpr(self):
        ''' fall-out or false positive rate (FPR)
            FPR = FP/N = FP/(FP + TN) = 1 − TNR'''
        licznik = self.tf.loc['FP',:]
        mian = self.tf.loc[['FP','TN'],:].sum(axis=0)  
        return licznik/mian


    def _x1fdr(self):
        ''' false discovery rate (FDR)
            FDR = FP/(FP + TP) = 1 − PPV '''
        licznik = self.tf.loc['FP',:]
        mian = self.tf.loc[['FP','TP'],:].sum(axis=0)  
        return licznik/mian


    def _x1foRate(self):
        ''' false omission rate (FOR)
            FOR = FN/(FN + TN) = 1 − NPV '''
        licznik = self.tf.loc['FN',:]
        mian = self.tf.loc[['FN','TN'],:].sum(axis=0)
        return licznik/mian


    def _x1ts(self):
        ''' Threat score (TS) or critical success index (CSI)
            TS = TP/(TP + FN + FP) '''
        licznik = self.tf.loc['TP',:]
        mian = self.tf.loc[['TP','FN','FP'],:].sum(axis=0)
        return licznik/mian


    def _x1mcc(self):
        ''' Matthews correlation coefficient (MCC)
            mcc = (TP*TN - FP*FN) / [(TP+FP) * (TP+FN) * (TN+FP) * (TN+FN)]^0.5
        '''        
        tp = self.tf.loc['TP',:]
        tn = self.tf.loc['TN',:]
        fp = self.tf.loc['FP',:]
        fn = self.tf.loc['FN',:]
        
        licznik = (tp * tn) - (fp * fn)
        mian = ((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))**0.5
        return licznik/mian

    # -----------------------------------------------------------------
    # wskaźniki złożone - modern2. Są obliczne na podstawie
    # wskaźników z grupy 1 - modern1.
    
    def _x2pt(self):
        ''' Prevalence Threshold (PT)
            PT = {[TPR*(1 − TNR)]^0.5 + TNR − 1} / (TPR + TNR − 1) '''
        licznik = (self.tpr * (1 - self.tnr))**0.5 +(self.tnr - 1)
        #mian = self.tf.loc[:,['tpr','tnr']].sum(axis=1)-1
        mian =self.tpr + self.tnr-1
        return licznik/mian


    def _x2ba(self):
        ''' Balanced accuracy (BA):
            ba = (TPR + TNR)/2
        '''
        return (self.tpr + self.tnr)/2


    def _x2f1(self):
        ''' F1 score is the harmonic mean of precision and sensitivity
            f1 = 2*(PPV*TPR)/(PPV+TPR) = (2*TP)/(2*TP+FP+FN)
        '''
        licznik = 2 * self.ppv * self.tpr
        mian = self.ppv + self.tpr
        return licznik/mian

    
    def _x2fm(self):
        ''' Fowlkes–Mallows index (FM)
            fm = [(TP/(TP+FP))*(TP/(TP+FN))]^0.5 = (PPV * TPR)^0.5
        '''
        return (self.ppv * self.tpr)**0.5

    
    def _x2bm(self):
        ''' informedness or Fowlkes–Mallows index (BM)
            bm = TPR + TNR - 1
        '''
        return self.tpr + self.tnr - 1

    
    def _x2mk(self):
        ''' markedness (MK) or deltaP
            mk = PPV + NPV - 1
        '''
        return self.ppv + self.npv - 1



#########################################################################################################    
#########################################################################################################


if __name__ == '__main__':
    # ustawienia
    np.set_printoptions(precision=9,linewidth=150)
    pd.set_option('expand_frame_repr', True)
    pd.set_option('precision', 9)
    prec = 5
    
    # 1. Testowa cross matrix
    nazwy = ['water','forest','urban'] # nazwy kolumn i wierszy (są takie same)
    value = [[21,5,7],[6,31,2],[0,1,22]] # liczby w komórkach
    cros = pd.DataFrame(value,columns=nazwy,index=nazwy)
    cros.axes[0].name='referencje'
    cros.axes[1].name='predicted'
    print(f'''\n1. Test klasy 'accClasic'Testowa!\n\n\tTestowa Coss matrix:\n\n{cros}\n\n''')
    
    # inicjacja instancji klasy
    cl = accClasic(cros,prec)
    
    w = int(round(0.7789474,cl.precision) * 10**cl.precision)
    t = int(cl.accOver * 10**cl.precision)
    assert t == w, f'''\n\nBłąd klasy 'accClasic', metody 'accOver': {t} != {w}\n'''
    
    
    w = np.round(np.array([0.63636364, 0.79487179, 0.95652174]), cl.precision)
    t = cl.accProd.to_numpy()
    assert np.all(t==w), f'''\n\nBłąd klasy 'accClasic', metody 'accPod'.\n'''
    
    
    w = np.round(np.array([0.77777778, 0.83783784, 0.70967742]), cl.precision)
    t = cl.accUser.to_numpy()
    assert np.all(t==w), f'''\n\nBłąd klasy 'accClasic', metody 'accUser'.\n'''   


    w = np.round(np.array([0.36363636, 0.20512821, 0.04347826]), cl.precision)
    t = cl.erOm.to_numpy()
    assert np.all(t==w), f'''\n\nBłąd klasy 'accClasic', metody 'erOm'.\n'''
    
    
    w = np.round(np.array([0.22222222, 0.16216216, 0.29032258]), cl.precision)
    t = cl.erCom.to_numpy()
    assert np.all(t==w), f'''\n\nBłąd klasy 'accClasic', metody 'erOm'.\n'''    

    print('''\n......... Wszystkie testy klasy 'accClasic' pozytywne - OK!\n\n''')

    #==============================================================================
    # testowa binTF tabela true/false
    
    binTF = [[21., 31., 22.],[56., 50., 63.],[ 6.,  6.,  9.],[12.,  8.,  1.]]
    binTF = pd.DataFrame(binTF, index=['TP', 'TN', 'FP', 'FN'], columns=['water', 'forest', 'urban'])
    print(f'''\n\n__2. Testy klasy 'accIndex'!\n\n\t2.1.Testowa tabela 'true/false':\n{binTF}\n\n''')
    
    # prawidłowe odpowiedzi 'indeksów' modern1
    tr = pd.DataFrame(
      [[0.810526316, 0.777777778, 0.636363636, 0.903225806, 0.823529412, 0.363636364, 0.096774194, 0.222222222,\
        0.176470588, 0.538461538, 0.569613037],
       [0.852631579, 0.837837838, 0.794871795, 0.892857143, 0.862068966, 0.205128205, 0.107142857, 0.162162162,\
        0.137931034, 0.688888889, 0.693791152],
       [0.894736842, 0.709677419, 0.956521739, 0.875      , 0.984375   , 0.043478261, 0.125      , 0.290322581,\
        0.015625   , 0.6875     , 0.759683931]])
       
    tr.columns = ['acc', 'ppv', 'tpr','tnr','npv','fnr','fpr','fdr','foRate','ts','mcc']
    tr.index = ['water', 'forest', 'urban']
    tr = np.round(tr,prec)
    
    print(f'''\n\t2.2.Tabela poprawnych wyników dla indeksów 'modern1':\n{tr}\n\n''')
    
    # incjacja instancji klasy
    cl = accIndex(binTF,precision=prec)
    
    for k,v in sorted(vars(cl).items()):
        if k in tr.columns.tolist():
            # wartości zamieniane są na 'int' inaczej wychodzi złe porównanie
            t = (tr.loc[:,k].to_numpy()*10**cl.precision).tolist()
            t = [int(round(x,0)) for x in t]
            
            w = (v.to_numpy()*10**cl.precision).tolist()
            w = [int(round(x,0)) for x in w]

            assert np.all(t==w), f'''\n\nBłąd klasy 'accIndex', zły index '{k}'\nt: {t}\n\nw: {w}\n'''

    print('''\n......... Testy klasy 'accClasic' w zakresie 'modern1' pozytywne - OK!\n\n''')
    
    #=================================================================================

 # prawidłowe odpowiedzi 'indeksów' modern2
    tr = pd.DataFrame(
      [[0.28055,     0.7698,      0.7        , 0.703526471, 0.539589442, 0.60130719 ],
       [0.26854,     0.843864469, 0.815789474, 0.816072096, 0.68773, 0.699906804],
       [0.265515653, 0.91576087 , 0.81482,     0.823906475, 0.831521739, 0.69406]])
    tr.columns = ['pt', 'ba', 'f1', 'fm', 'bm', 'mk']
    tr.index = ['water', 'forest', 'urban']
    tr = np.round(tr,cl.precision)
    print(f'''\n\t2.3.Tabela poprawnych wyników dla indeksów 'modern2':\n{tr}\n\n''')
    
    # incjacja instancji klasy
    #cl = accIndex(binTF,precision=prec)
    
    for k,v in sorted(vars(cl).items()):
        if k in tr.columns.tolist():
            # wartości zamieniane są na 'int' inaczej wychodzi złe porównanie
            #t = (np.round(tr.loc[:,k].to_numpy(), cl.precision)*10**cl.precision).astype('int')
            t = (tr.loc[:,k].to_numpy()*10**cl.precision).tolist()
            t = [int(round(x,0)) for x in t]
            
            w = (v.to_numpy()*10**cl.precision).tolist()
            w = [int(round(x,0)) for x in w]

            assert np.all(t==w), f'''\n\nBłąd klasy 'accIndex', zły index '{k}'\nt: {t}\n\nw: {w}
            \n{tr.loc[:,k]}'''



    print(f'''\n\n\n\t2.2.Testowa tabela 'true/false' dla indeksów 2 grupy:\n{binTF}\n\n''')

    print('\n  Wszystkie testy OK!\n')

    
