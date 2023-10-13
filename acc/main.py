# -*- coding: utf-8 -*-

from pathlib import Path

# my modules import
from simpleutils.src import verbosepk as vb

# local imports
from acc.src import args as parser  # parser argumentów w osobnym pliku
from acc.src.raport_base import SimpleRaport


# -----------------------------------------------------------------------------

def przygotujDaneInput(args):
    '''Funkcja przetwarza argumenty wejściowe skryptu'''
    # ustal katalog z którego uruchamiany jest skrypt
    args.runDir = Path(__file__).parent.resolve()

    if args.input:
        args.input = Path(args.input).resolve()
        args.work_dir = args.input.parent.resolve()
        args.input = args.input.as_posix()

    if args.raport:
        name = Path(args.input).name.split('.')[0]
        name = f'{name}_raport.html'
        args.raport = Path(args.input).with_name(name).as_posix()

    if args.ref is not None:
        args.ref = Path(args.ref).resolve().as_posix()

    return args


# -----------------------------------------------------------------------------

class AccRaport(SimpleRaport):

    def _konvertujDane(self):
        ''' Dane w tym skrypcie to pd.DataFrame. Przeciążenie oryginalnej
        metody polega na konwersji dataFrame do html-a.
        Input:
            - self.data:    lista danych, lista [pd.DataFrame, pd.DataFrame,..]
        '''
        htmlData = [dane.to_html() for dane in self.data]
        dataToRender = dict(zip(self.opis, htmlData))
        return dataToRender
# ---


def main():

    # =========================================================================
    # 1. Parsowanie argumentów wejścia
    # =========================================================================
    args = parser.parsuj_argumenty()
    args = parser.validuj_args(args)
    vb.verbose(pars=True, **vars(args))

# #############################################################################


if __name__ == '__main__':
    main()
    wykaz = None
    # =========================================================================
    # 2. Odczyt danych wejściowych
    # =========================================================================

#    if args.dataType == 'data':
#        data = pd.read_csv(args.input, sep=args.sep)
#        # ref  = pd.read_csv(args.ref, sep=args.sep)
#
#        cr = ConfusionMatrix(data)
#        cros = cr.cros
#        wykaz = ['data', 'cros']
#        if args.ref is not None:
#            ref = pd.read_csv(args.ref, sep=args.sep)
#            crosFull = OpisDlaConfMatrix(cr.cros1, ref).crosFull
#            wykaz.extend(['ref', 'crosFull'])
#
#        bin = BinTFtable(cros)
#        binTF = bin.binTF
#        wykaz.append('binTF')
#
#        # wykaz =['data','ref','cros','crosFull','binTF']
#        toSave = wykaz[1:]
#
#    elif args.dataType == 'cros':
#        cros = pd.read_csv(args.input, sep=args.sep, index_col=0)
#        # print(f'\n\n\ntuuuuuuuuuuuuuu:\n\n{cros}\n\n\n')
#        # to trzeba zapisać jako funkcje lub dodać do klasy crossMatrix!!!
#        # dodaje sumy wierszy i kolumn
#        cros1 = cros.copy()
#        sumRow = cros1.sum(axis=1).to_numpy()
#        cros1.loc[:, 'sumRow'] = sumRow
#
#        sumKol = cros1.sum(axis=0).to_numpy()
#        cros1.loc['sumCol', :] = sumKol
#
#        cros1 = cros1.astype('int')
#        # ---------------------------------------------------------------------
#
#        if args.ref is not None:
#            ref = pd.read_csv(args.ref, sep=args.sep)
#            crosFull = OpisDlaConfMatrix(cros1, ref).crosFull
#            wykaz = ['ref', 'crosFull']
#
#        bin = BinTFtable(cros)
#        binTF = bin.binTF
#        if wykaz is None:
#            wykaz = ['cros', 'binTF']
#        else:
#            wykaz.insert(0, 'cros')
#            wykaz .append('binTF')
#
#        toSave = ['cros', 'crosFull', 'binTF']
#
#    else:
#        binTF = pd.read_csv(args.input, sep=args.sep, index_col=0)
#        # sprawdź w jakim układzie jest data DataFrame
#        # print(f'\ntuu:\n{binTF}\n\n')
#        kols = set(binTF.columns.to_list())
#        spr = set(['TP', 'TN', 'FP', 'FN'])
#        if spr.issubset(kols):
#            binTF = binTF.T
#            # print(f'\ntuu:\n{binTF}\n\n')
#
#        wykaz = ['binTF']
#
#        # w układzie pionowym - na potrzeby raportu
#        binTFv = binTF.T
#        toSave = ['binTFv']
#        cros = None
#
#    vb.verbose(verbose=args.verbose, wykaz=wykaz)
#    # if args.verbose:
#    #     print('2. Dane wejściowe:\n')
#    #     print(f'wykaz: {wykaz}\n\n')
#    #     for it in wykaz:
#    #         print(f'''{it}:\n{eval(it)}\n\n''')
#
#    # =========================================================================
#    # 3. Tradycyjne, klasyczne wskaźniki dokładności
#    # =========================================================================
#
#    if cros is not None:
#        acc1 = accClasic(cros, args.precision)
#        print(100*'x')
#
#    else:
#        acc1 = accClasicBin(binTF, args.precision)
#
#    classicAcc = acc1.tabela
#    vb.verbose(verbose=args.verbose,
#               classicAcc=classicAcc)
#
#    toSave.append('classicAcc')
#    # ================================================================================
#    # 4. Nowe wskaźniki dokładności
#    # ================================================================================
#    acc2 = accIndex(binTF, precision=args.precision)
#    modern1 = {}
#    modern2 = {}
#
#    for k, v in vars(acc2).items():
#        if k in ['acc', 'ppv', 'tpr', 'tnr', 'npv', 'fnr', 'fpr', 'fdr',
#                 'foRate', 'ts', 'mcc']:
#            modern1[k] = v
#        elif k in ['pt', 'ba', 'f1', 'fm', 'bm', 'mk']:
#            modern2[k] = v
#
#    modern1 = pd.DataFrame(modern1)
#    modern2 = pd.DataFrame(modern2)
#
#    toSave.extend(['modern1', 'modern2'])
#
#    vb.verbose(verbose=args.verbose,
#               modern1=modern1, modern2=modern2)
#
#    # =========================================================================
#    # 4.1. Liczy średnie wartości wskaźników 'modern1' i 'modern2'
#    # =========================================================================
#
#    m1 = np.round(modern1.mean(), 4)
#    m2 = np.round(modern2.mean(), 4)
#
#    modernMean = pd.DataFrame(pd.concat([m1, m2]))
#
#    modernMean.columns = ['Value']
#
#    toSave.extend(['modernMean'])
#
#    vb.verbose(verbose=args.verbose, modernMean=modernMean)
#    # =========================================================================
#    # 5. Zapisywanie danych
#    # =========================================================================
#
#    if args.save and args.fullSave:
#        vb.verbose(verbose=args.verbose,
#                   zapis="""\t6.1. Polecenia zapisywania danych:\n""")
#
#        for nazwa in toSave:
#            if nazwa == 'binTF':
#                binTF = binTF.T
#            ad = Path(args.work_dir).joinpath(f'res_{nazwa}.csv')
#            polecenie = f'''{nazwa}.to_csv('{ad}',sep='{args.sep}')'''
#            vb.verbose(verbose=args.verbose, polecenie=polecenie)
#            eval(polecenie)
#
#    # tworzy raport html
#    if args.raport and args.save:
#        vb.verbose(verbose=args.verbose,
#                   polecenie='''\n\n\t6.2. Polecenia tworzenia raportu:\n''')
#
#        dane = [eval(nazwa) for nazwa in toSave]
#        raport = AccRaport(data=dane, opis=toSave)
#        raport.saveRaport(raportAdres=args.raport)
#
#    else:
#        print('''
#              Nie podano opcji '-s' (save) - raport nie zostanie wykonany!
#              \n''')
#
#    print('\n\n......Skrytp zakończony pozytywnie:\n\n')
