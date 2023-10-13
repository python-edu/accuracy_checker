# -*- coding: utf-8 -*-

import jinja2
from pathlib import Path

# --


wersja = 'w1.2020.08.13'
opis = '''
Wersja skryptu: {0}.

Moduł zawiera klasy dostarczające metody do wykonania prostego raportu z
obliczeń dokładności.
Klasy wykorzystywane w skryptach typu 'accuracy.py'

Klasy:
  ## 1. ...
  ## 2. ...

'''.format(wersja)


# -----------------------------------------------------------------------------

class SimpleRaport:
    ''' Klasa bazowa dla sporządzania prostego raportu przez wstawianie danych
    do formularza. W formularzu jest pętla 'for' która wstawia wszytsko co jest
    na liście danych.

        Dane składają się z dwóch list:
             - data:   lista tabeli do wstawienia np.: pd.DataFrame
             - opis:   tytuły tabel
             - przykład:
                    + data = [daneNr1, daneNr2, inneD]
                    + opis = ['Tytuł 1', 'Tytuł 2', 'Tytuł 3']

        Klasa wymaga zaimplementowania motody '_konvertujDane()' dokonującej
        konwersji surowych danych do odpowiedniego formatu np.
        - dla raportu 'html' konwersja do kodu html
        - dla raportu latex konwersja do latexh itd.
        wbudowaną metodą konwersji do html: pd.DataFrame.to__html()!!!.
    '''

    _defaultTemplate = '''<!DOCTYPE html >
    <html>
    <body>
       <div style="padding: 5px; width: 99%; background-color:rgb(165, 170, 175);">

           <div style="clear: both"><hr></div>

                {% for key,val in data.items() %}
                <div style="padding: 5px; width: 97%; background-color:rgb(195, 200, 205); margin-right: auto;
                  margin-left: auto;">
                <h2>{{key}}</h2>
                {{val}}</div>
                {% endfor %}

           <div style="clear: both"><hr></div>

       </div>
    </body>
    </html>
    '''

    def __init__(self,
                 data,
                 opis=['Dane'],
                 templateFolder=None,
                 templateName=None):
        ''' Args:
              - data:             list, lista danych np. lista dataFRame,
              - opis:             lis, lista opisów - nagłówków nad danymi
              - templateFolder:   adres katalogu z formularzami
              - template:         nazwa pliku formularza - plik musi
                                  znajdować się w katalogu 'templateFolder'
              - out:              str, adres pliku raportu
        '''
        self.raport = None

        # ile danych do raportu
        self._n = len(data)
        self.templateName = templateName
        self._env, self.templateFolder = self._setupRaport(templateFolder)
        self.template = self._getTemplate()

        self.opis = opis
        self.data = data

    # ---

    def _setupRaport(self, templateFolder):
        env = None
        if templateFolder is not None:
            templateFolder = Path(templateFolder).resolve().as_posix()
            tmpLoader = jinja2.FileSystemLoader(templateFolder)
            env = jinja2.Environment(loader=tmpLoader)
        return env, templateFolder
    # ---

    def _getTemplate(self):
        template = None
        if self.templateName is not None:
            try:
                template = self._env.get_template(self.templateName)
            except jinja2.exceptions.TemplateNotFound:
                print(f'''\n\n\tRaport nie został utworzony!!!

                    Formularz raportu, plik '{self.templateName}',
                    nie został znaleziony w podanym katalogu!!!\n
                    Do raportu wykorzystany zostanie wbudowany szablon!!\n\n
                      ''')
                template = jinja2.Template(self._defaultTemplate)
        else:
            template = jinja2.Template(self._defaultTemplate)

        return template

    def _konvertujDane(self):
        dataToRender = dict(zip(self.opis, self.data))
        return dataToRender
    # ---

    def _raport(self):
        return self.template.render(data=self._dataToRender)
    # ---

    def saveRaport(self, raportAdres):
        with open(raportAdres, 'w') as f:
            f.write(self.raport)
    # ---

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self._dataToRender = self._konvertujDane()
        self.raport = self._raport()
    # ---

    @property
    def opis(self):
        return self._opis

    @opis.setter
    def opis(self, opis):
        ''' Opis powinien być listą nazw tabel z danymi
        '''
        if not isinstance(opis, list):
            msg = 'Ostrzeżenie: opis danych powinien być listą z nazwami tabel'
            print(msg)
            opis = [opis]

        lenOp = len(opis)
        n = self._n

        if lenOp < n:
            ll = n - lenOp
            tmpOpis = [opis[-1] for i in range(ll)]
            opis.extend(tmpOpis)

            # liczba zer przed numerem
            if n < 100:
                nz = 2
            elif n >= 100 and n < 1000:
                nz = 3

            opis = [f'{x}_{i:0{nz}d}' for i, x in enumerate(opis, 1)]

        else:
            opis = opis[:]
        self._opis = opis

# #############################################################################


if __name__ == '__main__':

    dane = [f'same data {i}' for i in range(1, 4)]
    opisy = [f'opis{i}' for i in range(1, 4)]

    print(f'\n Dane do zapisu jako raport:\n{dane}\n\n')

    rap = SimpleRaport(dane, opisy)
    print(f'Raport:\n\n{rap.raport}\n\n')
