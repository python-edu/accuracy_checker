import re
import textwrap
import json
import base64
from collections import OrderedDict
from pathlib import Path
from importlib.resources import files


imgs_replacement = {
        'run1': '-:$ accuracy data2cols.csv',
        'run2': '-:$ accuracy data2cols.csv class_map.json',
        'run3': '-:$ accuracy data3cols.csv',
        'run4': '-:$ accuracy data3cols.csv class_map.json',
        'cross_raw1': '-:$ accuracy cross_raw.csw',
        'cross_raw2': '-:$ accuracy cross_raw.csw class_map.json',
        'cross1': '-:$ accuracy cross.csv',
        'cross2': '-:$ accuracy cross.csv class_map.json',
        'raster1': ('-:$ accuracy raster.tif  # this will generate an error: '
                    'no file `raster_ref`!!!'),
        'raster2': '-:$ accuracy raster.tif  # only if exists `raster_ref.tif`',
        'raster3': ('-:$ accuracy raster.tif class_map.json  # only if exists '
                    '`raster_ref.tif`'),
        'raster4': '-:$ accuracy raster.tif reference_vector.gpkg',
        'example_data': '-:$ accuracy example_data cross.csv class_map.json',
        }


# funkcje do przetwarzania tekstu pomocy dla streamlit (uproszczony
# markdown)

def mk_tables(txt: list[str]) -> list[str]:
    """Przetwarza wiersze zawierające tabele, czyli zaczynające się od `|`."""
    res = []
    txt = txt[:]

    # start: False - wiersz to nie tabela, True - to tabela
    start = False

    for line in txt:
        if line.startswith('|'):
            # początek tabeli
            if not start:
                start = True
                res.append("  ```bash")
                
            line = f"    {line}"
            res.append(line)
        else:
            # breakpoint()
            if start:
                start = False
                res.append("  ```")
            res.append(line)
    
    return res


def mk_equations(txt: list[str]) -> list[str]:
    """Przetwarza linie zawierające równania, czyli wiersze zaczynające
    się od `--`.
    """
    res = []
    txt = txt[:]

    for line in txt:
        if line.startswith('--'):
            line = line.replace('--', '')
            line = line.strip()
            line = '\n' + f"$$\n{line}\n$$"
            line = line.replace(r'^0.5', r'^{0.5}')
        res.append(line)
    return res


def get_index(txt: list[str], pattern: str) -> list[str]:
    """Przechodzi linia po linii i wyszukują tę linię która pasuje do
    wzorca. Zwraca index tej linii.
    """
    idx = None
    txt = txt[:]

    for i, line in enumerate(txt):
        if re.search(pattern, line):
            idx = i
    return idx


class Readme2HelpCli:
    """Klasa służy do konwersji pliku README.md na tekst pomocy wyświetlany
    przez skrypt w konsoli tekstowej.
    """
    chapters = {
            'usage_help': '# Usage help',
            'data_help': '# Data help',
            'metrics_help': '# Metrics help',
            'formula_help': '# Formula help'}

    patterns = OrderedDict({
        'h1': re.compile(r'^#\s+'),
        'h2': re.compile(r'^##\s+'),
        'h3': re.compile(r'^###\s+'),
        'empty': re.compile(r"^\s*$"),
        'hash': re.compile(r"^\s+#\s.+"),
        'bash': re.compile(r'.*```.*$'),
        'bold': re.compile(r'^\s{1,3}[*]{2}.*[*]{2}'),
        'lists': re.compile(r'^\s*-\s+|^\s*--\s|^\s*+\s'),
        'colon': re.compile(r':$'),
        'quote': re.compile(r'^\s*>\s*'),
        'table': re.compile(r'^\s*\|'),
        'img': re.compile(r"^\s*!\[(.+)\]"),
        'equation': re.compile(r"^\s*[$]{2}"),
        # 'usage_help': re.compile(r"^\s{3,}.+#.+"),
        'terminal': re.compile(r"^\s*-:[$]\s.+"),
        'json': re.compile(r'\s*{\s*$|\s*}\s*$'),
        })

    def __init__(self, txt, indent=4, width=110):
        """Args:
          - txt:  str, from README.md
          - indent:  int, ile spacji wcięcia całego tekstu
          - width: int, całkowita długość linii (szerokość tekstu)
        """
        # ustawia atrybuty dla instancji: klucze z chapters
        for key in type(self).chapters:
            setattr(self, key, None)
        self.patterns = type(self).patterns.copy()
        self.indent = indent
        self.width = width
        self.global_indent = 4

        self.txt = [line.rstrip() for line in txt.splitlines()]
        self._split_chapters(self.txt)

        for chapter_name in type(self).chapters:
            # numeracja nagłówków przetwarzanego README
            self.h2_idx = 0
            self.h3_idx = 1

            chapter_txt = getattr(self, chapter_name).splitlines()
            # groups = self._split_groups(getattr(self, chapter_name).splitlines())
            groups = self._split_groups(chapter_txt)
            help_txt = self._format_groups(groups)
            setattr(self, chapter_name, '\n'.join(help_txt))
        
        del self.h2_idx, self.h3_idx

    def _split_chapters(self, lines: list[str]) -> list[dict]:
        """Dzieli tekst z README.md na osobne rozdziały, zgodnie z atrybutem
        `chapters`.
        """
        # chapters: {'key': 'pattern'}
        chapters = type(self).chapters.copy()
        res = {key: [] for key in chapters.keys()}

        # reverse chapters {'pattern': 'key'}
        chapters = dict(zip(chapters.values(), chapters.keys()))
        curr_chapt = None
        
        for line in lines:
            # pomija linie '---'
            if re.search(r'---', line):
                continue

            # usuwa cytowania `>`
            line = line.replace('>', '', 1)

            if re.search(r"^#\s.+", line) and chapters.get(line, False):
                curr_chapt = chapters.get(line)
            elif re.search(r"^#\s.+", line):
                curr_chapt = None

            # jeśli aktualnie jest wykrywany jakiś chapter
            if curr_chapt is not None:
                # sprawdź czy to linia nie zaczyna nowego chaptera
                check = chapters.get(line, None)
                if check is None:
                    res[curr_chapt].append(line)
                    continue
                else:
                    curr_chapt = check
                    continue
        
        for chapter_name, lines in res.items():
            lines = '\n'.join(lines).strip()
            setattr(self, chapter_name, lines)
            
    def _split_groups(self, lines: list[str]) -> list[dict]:
        """Dzieli tekst na grupy (nagłówki, listy, ...) do których stosuje
        odpowiednie formatowanie tekstu.
        """
        res = []
        current_group = False 
        patterns = self.patterns.copy()
        pattern_json = patterns.pop('json')
        reverse_patterns = dict(zip(patterns.values(),
                                    patterns.keys())
                                )
    
        for line in lines:
            # 1. usuwan znaki `>` cytowania
            if re.search(r"^\s*[>]", line):
                line = line.replace(">", '', 1).strip()
            
            # 2. szuka json
            if pattern_json.search(line):
                # breakpoint()
                if current_group != 'json':
                    # rozpoczyna block json `{`
                    res.append({'json': line})
                    current_group = 'json'
                elif current_group == 'json':
                    # kończy blok json `}`
                    res[-1]['json'] += '\n' + line
                    current_group = False
                continue

            # jeśli json to znaczy że blok json jest otwarty i ma być
            # kontynuowany
            if current_group == 'json':
                res[-1]['json'] += '\n' + line
                continue

            # 3. Sprawdza dopasowania pojedynczych linii
            for pattern, name in reverse_patterns.items():
                if pattern.search(line):
                    if name == 'colon' and current_group == 'paragraf':
                        continue

                    current_group = False
                    if name == 'bash':
                        break

                    res.append({name: line})
                    break
            
            # dziwna wersja pętli `for`!!!
            # 4. Jeśli do tego momentu nie ma dopasowania -> paragraf
            else:
                if current_group == 'paragraf':
                    res[-1]['paragraf'] += ' ' + line.strip()
                elif current_group != 'paragraf':
                    current_group = 'paragraf'
                    res.append({'paragraf': line.strip()})

        return res

    def _format_groups(self, groups: list[dict]):  #, n, width):
        res = []
        # pamięta poprzednią grupę
        name_mem = None
        n = self.global_indent 

        for gr in groups:
            name = list(gr.keys())[0]
            method = getattr(self, f"_format_{name}")
            txt = method(gr[name], name=name_mem)
            if name != 'json':
                txt = f"{n * ' '}{txt}"

            try:
                res.append(txt)
            except Exception:
                print(f"\n\nError:\n{gr}\n\n")
                import sys
                sys.exit(1)
            name_mem = name
        res += '\n'
        return res

    def _format_h1(self, txt, **kwargs):
        # txt: '# Some title'
        txt = txt[1:].strip()
        return txt

    def _format_h2(self, txt, **kwargs):
        # txt: '## Some title'
        self.h2_idx += 1
        self.h3_idx = 1
        txt = txt[2:].strip()
        txt = f"{self.h2_idx}. {txt}"
        return txt

    def _format_h3(self, txt, **kwargs):
        # txt: '### Some title'
        txt = txt[3:].strip()
        txt = f"  {self.h2_idx}.{self.h3_idx}. {txt}"
        self.h3_idx += 1
        return txt

    def _format_img(self, line, **kwargs):
        key = self.patterns['img'].search(line).group(1)
        # `imgs_replacement`: zmienna globalna modułu
        # txt = f"{3*' '}{imgs_replacement[key]}"
        line = imgs_replacement[key]
        line = self._format_as_list(line)
        return line

    def _format_equation(self, line: str, **kwargs):
        name = kwargs.get('name')
        n = 4
        if name == 'lists':
            n=6
        line = f"{n*' '}{line.strip()}"
        return line

    def _format_bold(self, line: str, **kwargs):
        return line

    def _format_colon(self, line: str, **kwargs):
        return f" {line}"

    def _format_empty(self, txt, **kwargs):
        return ''

    def _format_as_list(self, line: str, **kwargs):
        """Formatuje dowolną linię jako listę: np.:
          - line = 'abc 345' -> '    - abc 345'
        """
        line = line.strip()
        width = self.width
        n = self.indent

        if line.startswith('-'):
            line = f"{n * ' '}{line}"
        else:
            line = f"{n * ' '}- {line}"

        line = textwrap.fill(line,
                             width=width,
                             subsequent_indent=(3 + n) * " "
                             )
        return line

    def _format_lists(self, line, **kwargs):
        return self._format_as_list(line)

    def _format_usage_help(self, txt, **kwargs):
        return self._format_lists(txt)

    def _format_terminal(self, line: str, **kwargs):
        return self._format_as_list(line)

    def _format_hash(self, line, **kwargs):
        line = line.replace('#', '')
        return self._format_as_list(line)

    def _format_paragraf(self, txt, **kwargs):  #, width):
        """Zwykły tekst wieloliniowy, składany jako akapit tekstu."""
        txt = [line.strip() for line in txt.splitlines()]
        txt = [" ".join(line.split()) for line in txt]
        txt = " ".join(txt)
        txt = textwrap.fill(txt,
                            width=self.width,
                            initial_indent=' ',
                            subsequent_indent=4 * " ")
        return txt

    def _format_table(self, table, **kwargs):  #, n):
        n = self.indent + 4
        table = [line.strip() for line in table.splitlines()]
        table = [f'{" "*n}{line}' for line in table]
        table = "\n".join(table)
        return table

    def _format_json(self, txt, **kwargs):
        txt = json.loads(txt)
        txt = json.dumps(txt, ensure_ascii=False, indent=2)
        n = self.global_indent + 4
        txt = textwrap.indent(txt, n*' ')
        return txt


class Readme2Streamlit(Readme2HelpCli):
    """Klasa dzieli tekst odczytany z pliku README.md na potrzeby wyświetlania
    w app.py.
    """
    ...

    # def __init__(self, txt, indent=4, width=110):
    def __init__(self, txt, docs_source):
        """Args:
            - txt:  str, text from README.md
            - docs_source:  importlib.resources , virtual path to folder with
              images (screeny z konsoli pokazujące przykłady użycia)
        """
        # ustawia atrybuty dla instancji: klucze z chapters
        for key in type(self).chapters:
            setattr(self, key, None)
        self.patterns = type(self).patterns.copy()

        self.txt = [line.rstrip() for line in txt.splitlines()]
        self._split_chapters(self.txt)
        self.docs_source = docs_source

        # dostosowuje metrics_help do streamlit
        self.metrics_help = self._format_metrics_help()

        # rozwiąż ścieżki do obrazków
        self.usage_help = self._resolve_img_paths()

    def _format_metrics_help(self):
        res = []
        txt = self.metrics_help.splitlines()
        pattern = self.patterns.get('equation')
        for line in txt:
            if pattern.search(line):
                line = '\n' + line
            res.append(line)
        res = '\n'.join(res)
        return res

    def _resolve_img_paths(self):
        res = []
        txt = self.usage_help.splitlines()
        # pattern = re.compile(r"^\s*!\[.+]\((.+)\)")
        pattern = re.compile(r"(^\s*!\[.+]\()(.+)(\))")

        for line in txt:
            if pattern.search(line):
                img_pth = Path(pattern.search(line).group(2))
                name = str(img_pth.name)
                img_source = self.docs_source.joinpath(name)
                if img_source.is_file():
                    data = base64.b64encode(img_source.read_bytes()).decode()
                    ext = img_pth.suffix.lstrip(".")
                    line = (f"<img alt='{name}' "
                            f"src='data:image/{ext};base64,{data}' "
                            " style='max-width:50%; border-radius:8px;'>")
                # line = pattern.sub(fr'\1{pth}\3', line)
                # print(line)
        
            res.append(line)
        res = '\n'.join(res)
        return res
        




def parse_help_text(txt):
    """Funkcja przetwarza tekst pomocy na markdown używany w streamlit."""
    txt = txt.splitlines()
    txt = [line.strip() for line in txt]
    if txt[0] == '':
        txt = txt[1:]

    txt = [f"#### {line}" if re.search(r'^\d\.\d', line) else
           line for line in txt]
    txt = [f"### {line}" if re.search(r'^\d', line) else line for line in txt]
    txt = ['\n' if line == '' else line for line in txt]

    # txt = [f"\t{line}" if line.startswith('|') else line for line in txt]
    txt = mk_tables(txt)
    
    # przetwarza równania (linie zaczynają si e od --) 
    txt = mk_equations(txt)
    txt = [f"  {line}" if line.startswith('-') else line for line in txt]


    # format jednej konkretnej linii
    # pat = "- '*.tif', '*.tiff', '*.TIF', '*.TIFF'"
    pat = r"\*\.TIFF"
    idx = get_index(txt, pat)
    if idx is not None:
        txt[idx] =  "\n  ```bash \n  #'*.tif', '*.tiff', '*.TIF', '*.TIFF'\n```"
    
    txt = '\n'.join(txt)
    return txt
