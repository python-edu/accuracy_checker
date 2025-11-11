# io, contextlib: w celu przekierowania strumienia stdout do streamlit
import io
import contextlib

from pathlib import Path
from types import SimpleNamespace

import streamlit as st
from streamlit_navigation_bar import st_navbar


# local imports
from acc.src.args_data import help_info
from acc.src.args_data import streamlit_args
from acc.src.args_data import args_func as afn
from acc.main import main


# setup for help
help_map = {"Usage": "usage_help",  # "info_usage",
            "About data": "data_help",
            "About metrics": "metrics_help",
            "Formula": "formula_help"}

st.markdown("""
<style>
/* --- Mniejsze przyciski (globalnie) --- */
.stButton > button {
    padding: 0.2rem 0.5rem !important;
    font-size: 0.8rem !important;
    min-height: 0 !important;
    min-width: 0 !important;
}

</style>
""", unsafe_allow_html=True)


# setup streamlite
st.set_page_config(page_title="Accuracy", layout="wide")
page = st_navbar(["Calculations", "Usage", "About data", "About metrics",
                  "Formula"], adjust=False)


# === GLOBALNE i funkcje / klasy używane w `Calculations` ===


def get_name(dir_path: str):
    """Funkcja wykorzystywana w selectbox(), dostarcza krótką nazwę
    pliku/katalogu w okienku wyboru pliku.
    """
    name = Path(dir_path).name
    return name


def setup_dirs():
    if "root" not in st.session_state:
        root = Path(__file__).parent.parent.resolve()
        st.session_state['root'] = root

    if 'readme_path' not in st.session_state:
        st.session_state['readme_path'] = st.session_state.root / 'README.md'

    if 'docs_path' not in st.session_state:
        st.session_state['docs_path'] = next(st.session_state.root.rglob('docs'))
    

    cwd = (Path(st.session_state.root) / "example").resolve()
    # globalna - referencja
    st.session_state.setdefault('CWD', cwd)

    if "cwd" not in st.session_state:
        # cwd = (Path(st.session_state.root) / "example").resolve()
        st.session_state["cwd"] = cwd

    # fdir - obsługuje wybór katalogów do wyboru plików
    if 'fdir' not in st.session_state:
        cwd = st.session_state.cwd
        st.session_state['fdir'] = streamlit_args.ListDirectories(root=cwd)
        st.session_state['list_dirs'] = st.session_state.fdir.list_dirs

    # ocwd i odir - obsługują wybór katalogów dla out_dir
    # out_dir wybiera użytkownik, więc startowo None
    st.session_state.setdefault('ocwd', None)
    if 'odir' not in st.session_state:
        cwd = st.session_state.cwd
        st.session_state['odir'] = streamlit_args.ListDirectories(root=cwd)


def dir_selection(which: str = 'path', disabled=False):
    """Umożliwia wybór katalogu:
        - na potrzeby wyboru plików (args.path)
        - na potrzeby wyboru katalogu do zapisu wyników (args.out_dir)
    
    Args:
      - which:  str, `path` lub `out_dir`, wskazuje czego dotyczyć będzie
                wybór katalogu, czy plików czy out_dir

    """
    if which == 'path':
        label = 'Data_dir'
        ld_key = 'fdir'
        cwd_key = 'cwd'
        select_key = 'for_path'
    elif which == 'out_dir':
        label = 'Out dir'
        ld_key = 'odir'
        cwd_key = 'ocwd'
        select_key = 'for_out_dir'

    # base_dir:
    #    to katalog bieżący, zależnie od wyboru `cwd` lob `ocwd`
    #    służy do ustalenia wartości obecnej `cwd` / `ocwd` i potem porównania
    #    tych wartości - jeśli nowe wartości `cwd` / `ocwd` są rózne od
    #    base_dir to następuje ich wprowadzenie (aktualizacja) do session_state

    if which == 'path':
        base_dir = st.session_state[cwd_key]

    # `out_dir` - to jest inny niż dla `path`:
    # - jeśli już istnieje session_state.ocwd to je ustawia jako base_dir
    # - jeśli nie to próbuje wykorzystać katalog pierwszego pliku
    # - jeśli nie jest jeszcze ustawiony to pierwszą wartością jest katalog
    #   `cwd`, który zawsze istnieje bo ma wartość domyślną
    else:
        if st.session_state[cwd_key]:  # tu cwd_key = `ocwd`
            base_dir = st.session_state[cwd_key]
        elif st.session_state.pth1:
            base_dir = st.session_state.pth1.parent
        else:
            # żeby nie był pusty to ostatecznie taki sam jak obecny `cwd`
            base_dir = st.session_state['cwd'] 


    # disabled == False (przycisk włączony)
    # - dla which == `path` - zawsze
    # - dla which == `out_dir` - tylko czasem
    if not disabled:
        dirr = st.session_state[ld_key]
        dirr(base_dir)
        list_dirs = dirr.list_dirs

    # disabled == True (przycisk nieaktywny)
    # - dla which == `out_dir` - gdy nie wybrano żadnej z opcji zapisu danych
    #   np. save, zip, report
    else:
        list_dirs = []

    # which == `out_dir`: label będzie wyświetlać aktualną wartość `ocwd`
    #  - czyli `base_dir`
    if which == 'out_dir':
        label = f'{label}: {base_dir}'

        # DODANE: domyślna selekcja na base_dir po włączeniu wyboru
        if (not disabled) and base_dir:
            # ustaw stan selectboxa na base_dir (jeśli jeszcze pusty)
            if st.session_state.get('for_out_dir') is None and base_dir in list_dirs:
                st.session_state['for_out_dir'] = base_dir
            # ustaw ocwd i args['out_dir'], jeśli jeszcze puste
            if st.session_state.get('ocwd') is None:
                st.session_state['ocwd'] = base_dir
            if st.session_state.args.get('out_dir') is None:
                st.session_state.args['out_dir'] = base_dir

    new_folder = st.selectbox(label,
                           options=list_dirs,
                           format_func=get_name,
                           index=None,
                           placeholder="— Select folder —",
                           disabled=disabled,
                           key=select_key)

    if new_folder is not None:
        new_folder = Path(new_folder).resolve()
        if new_folder != base_dir:
            if which == 'out_dir':
                st.session_state.args['out_dir'] = new_folder
            st.session_state[cwd_key] = new_folder


def format_paths(paths_source=False, null='---', level=3) -> list[str]:
    """Zwraca skrócone ścieżki do plików, na potrzeby wyświetlania informacji
    o nich."""
    paths = []
   
    # normalizacja źródeł do słownika
    # breakpoint()
    if not paths_source:
        paths_source = {f'pth{i}': st.session_state[f'pth{i}'] for i in (1,2,3)}

    elif isinstance(paths_source, str|Path):
        paths_source = {'pth1': str(paths_source)}

    elif isinstance(paths_source, list):
        p_range = list(range(1, len(paths_source)+1))
        # breakpoint()
        paths_source = {f'pth{i}': str(paths_source[i-1]) for i in p_range}

    for pth in ('pth1', 'pth2', 'pth3'):
        # pth = st.session_state[pth]
        pth = paths_source.get(pth)
        if pth is not None:
            pth = Path(pth)
            parents = pth.parents
            if level >= len(parents):
                n = len(parents) - 1
            else:
                n = level
            new_pth = Path(pth).relative_to(parents[n])

            if n == level:
                new_pth = f".../{new_pth}"
            paths.append(str(new_pth))
        else:
            paths.append(null)

    return paths


def update_args():
    st.session_state.args['path'] = []
    # res = ' '
    # res = []
    for name in ('pth1', 'pth2', 'pth3'):
        pth = st.session_state.get(name, False)
        if pth:
            pth = (str(pth)).strip()
            st.session_state.args['path'].append(pth)
            # res += f'{str(pth)} '
    # res = res.strip()
    # st.session_state.args['path'] = res


class _WriteToStreamlit(io.TextIOBase):
    """Klasa przekierowuje komunikaty ze skryptu - main(args), które normalnie
    są wyświetlane w konsoli do pustego pojemnika (box) w streamlit -
    st.empty().

    Klasa dziedziczy obiekt `plikopodobny` z modułu `io` i wymaga napisania
    przynajmniej metody `write()` bo taką musi mieć każdy taki obiekt.

    """

    def __init__(self, box):
        self.box = box
        self.bufor = []  # pojemnik na wiadomości z main()

    def write(self, msg):
        """msg - z założenia obiekt typu str"""
        if not isinstance(msg, str):
            msg = msg.decode("utf-8", "ignore")

        self.bufor.append(msg)
        self.box.code("".join(self.bufor))
        return len(msg)


def del_path(file_key: str):
    """Usuwa pojedynczy wybór pliku z session_state.args i session_state.pth...
    W przypadku File_1 usuwa również out_dir.
    Args:
      - file_key:  str, one of [File_1, File_2, File_3]
    """
    if file_key == 'File_1':
        pth = 'pth1'
    elif file_key == 'File_2':
        pth = 'pth2'
    elif file_key == 'File_3':
        pth = 'pth3'

    # KLUCZ SELECTBOXA, który zapamiętuje poprzedni wybór:
    sel_key = f"select_file_{file_key}"

    # wyczyść UI-select, żeby wrócił placeholder („— Select file —”)
    st.session_state[sel_key] = None
    
    # wyczyść session_state
    st.session_state[pth] = None
    if file_key == 'File_1':
        st.session_state.args['out_dir'] = None  #st.session_state.cwd


def del_all_path():
    """Usuwa wszystkie wybrane pliki z session_state.args i session_state.pth...
      - [File_1, File_2, File_3]
    """
    for file_key in ('File_3', 'File_2', 'File_1'):
        del_path(file_key)


# === start program ===========================================================

# to wyświetla pomoc
if page != "Calculations":
    # --- help section
    st.write(f"## {page}")
    text_name = help_map.get(page, 'no')
    with open(st.session_state['readme_path']) as f:
        readme_txt = f.read()
    help_txt = help_info.Readme2Streamlit(readme_txt,
                                          st.session_state.docs_path)
    # help_text = getattr(help_info, text_name, "No help")
    # help_text = help_info.parse_help_text(help_text)
    st.markdown(getattr(help_txt, text_name), unsafe_allow_html=True)


# to uruchamia obliczenia - skrypt `main.py`
if page == "Calculations":
    report_data = ['title=Image Classification Accuracy Assessment',
                   'description=Research project.',
                   'report_file=report.html',
                   'template_file=report_template.html',
                   'template_dir=templates']

    # nazwy argumentów i wartości domyślne
    args = {'path': [],
            "save": False,
            "out_dir": None,
            "report": False,
            "report_data": report_data[:],
            "precision": 4,
            "zip": False,
            "zip_name": None,
            "formula": None,
            "sep": None,
            "reversed": False,
            "verbose": False
            }

    file_pattern = {'File_1': ['*.csv', '*.tif', '*.tiff', '*.TIF', '*.TIFF'],
                    'File_2': ['*.tif', '*.tiff', '*.TIF', '*.TIFF',
                               '*.shp', '*.gpkg'],
                    'File_3': ['*.json']
                    }
    file_name = {'File_1': 'pth1',
                 'File_2': 'pth2',
                 'File_3': 'pth3' 
                    }
    # --- setup
    setup_dirs()
    st.session_state.setdefault('args', args)

    for pth in ('pth1', 'pth2', 'pth3'):
        st.session_state.setdefault(pth, None)

    st.session_state.setdefault('choice', 'File_1')

    # --- pasek boczny
    with st.sidebar:
        st.markdown("# Positional arguments:")
        col_ch, col_del, col_del_all = st.columns([4, 1, 1])
        
        with col_ch.container():
            choice = st.radio("File no",
                              options=("File_1", "File_2", "File_3"),
                              index=0,
                              horizontal=True,  # w jednej linii
                              key="file_selector",
                              label_visibility="collapsed"
                              )

        # choice: File_1, File_2, File_3
        if choice != st.session_state.choice:
            st.session_state.choice = choice

        # usuwanie wybranych plików z st.session_state
        with col_del.container():
            # wywołuje funkcję `del_path`
            st.button('Del path',
                      key='del_single',
                      on_click=del_path,
                      args=[choice])
        
        with col_del_all.container():
            # wywołuje funkcję `del_all_paths`
            st.button('Del all',
                      key='del_all_paths',
                      on_click=del_all_path,
                      )

        # sekcja wyboru plików: informacja o aktualnym katalogu `cwd`
        #   - nie dotyczy sekcji wyboru `out_dir`!!!
        cwd = st.session_state.cwd
        if cwd == Path.home():
            cwd_str = '~/'
        else:
            cwd_str = f"~/.../{cwd.relative_to(Path.home())}"
        st.markdown(f"> `Current folder`: "
                    f"&emsp;**{cwd_str}**")
        
        c1, c2, c3, c4 = st.columns([5,6,1,1])
        c_dir = c1.container()   # select directory
        c_file = c2.container()  # select file
        c_up = c3.container()    # button up folder
        c_home = c4.container()  # button set home folder

        # --- sekcja wyboru katalogu `cwd` i plików file1, file2, file3
        with c_dir.container():
            dir_selection('path') 

        with c_file.container():
            cwd = st.session_state.cwd
            patterns = file_pattern[st.session_state.choice]
            file_list = [pth for pat in patterns for pth in cwd.glob(pat)]
            file_list = sorted(file_list)
            new_pth = st.selectbox(f'Select {choice}:',
                                   options = file_list,
                                   format_func=get_name,
                                   index=None,
                                   placeholder="— Select file —",
                                   key=f'select_file_{choice}')

            # ustala którego pliku dotyczy wybór: pth1, pth2, pth3
            name = file_name[choice]

            if new_pth is not None:
                new_pth = Path(new_pth).resolve()
                if new_pth != st.session_state[name]:
                    st.session_state[name] = new_pth
                    # st.rerun()

        # --- pokazuje aktualnie wybrane pliki
        paths_repr = format_paths()
        st.markdown(f">  - `file1`:&emsp;**{paths_repr[0]}**  \n"
                    f">  - `file2`:&emsp;**{paths_repr[1]}**  \n"
                    f">  - `file3`:&emsp;**{paths_repr[2]}**  \n"
                    "---")


        # --- przyciski up i home
        with c_up.container():
            # on_off: kiedy cwd == home to wyłącza przycisk, żeby nie było
            # próby wychodzenia poza katalog domowy i eliminuje błąd
            on_off = cwd == Path.home() 
            if st.button("\u2191",
                         disabled=on_off,
                         width="stretch",
                         key='up_butt'):
                st.session_state["cwd"] = cwd.parent.resolve()
                # czyści selectbox wyboru katalogu -> --Separator folder--
                # st.session_state['for_path'] = None
                st.rerun()
        
        with c_home.container():
            if st.button("~", disabled=False, width="stretch", key='home_butt'):
                st.session_state["cwd"] = Path.home()
                # czyści selectbox wyboru katalogu -> --Separator folder--
                # st.session_state['for_path'] = None
                st.rerun()


        # === sekcja other options ============================================
        # zaznaczanie wyboru: save, zip, report, reversed
        st.markdown("# Optional arguments:")
        a11, a12, a13 = st.columns([1.5, 1, 1])
        save_zip = a11.radio('save options',
                             options=['no', 'save', 'zip'],
                             index=0,
                             horizontal=True,
                             label_visibility="collapsed")

        # --- można wybrać tylko jedną z 3 opcji: no, save, zip
        #     wybór będzie blokował inne możliwości
        if save_zip == 'no':
            st.session_state.args['save'] = False
            st.session_state.args['zip'] = False
        elif save_zip == 'save':
            st.session_state.args['save'] = True 
            st.session_state.args['zip'] = False
        elif save_zip == 'zip':
            st.session_state.args['save'] = False
            st.session_state.args['zip'] = True

        st.session_state.args['report'] = a12.checkbox('Report')
        st.session_state.args['reversed'] = a13.checkbox('Reversed')

        # === przyciski wyboru katalogu ===
        a21, up1, c_home1 = st.columns([6, 1, 1])
        
        with up1.container():
            # on_off: kiedy cwd == home to wyłącza przycisk, żeby nie było
            # próby wychodzenia poza katalog domowy i eliminuje błąd
            ocwd = st.session_state.ocwd
            on_off = ocwd == Path.home() 
            if st.button("\u2191",
                         disabled=on_off,
                         width="stretch",
                         key='up_butt1'):
                st.session_state["ocwd"] = cwd.parent.resolve()
                # czyści selectbox wyboru katalogu -> --Separator folder--
                st.session_state['for_out_dir'] = None
                st.rerun()
        
        with c_home1.container():
            if st.button("~",
                         disabled=False,
                         width="stretch",
                         key='home_butt1'):
                st.session_state["ocwd"] = Path.home()
                # czyści selectbox wyboru katalogu -> --Separator folder--
                st.session_state['for_out_dir'] = None
                st.rerun()

        # wybór katalogu `out_dir`
        with a21.container():
            disabled = not (st.session_state.args['save'] or
                            st.session_state.args['zip'] or
                            st.session_state.args['report'])
            dir_selection('out_dir', disabled=disabled)


        # === przyciski: precision i formula 
        sep_csv, prec_check, formula_check, verbose_col = st.columns(4)

        # separator w plikach csv
        with sep_csv.container():
            mark_sep = st.checkbox('Separator', key='mark_sep')
            if mark_sep:
                sep = st.selectbox('csv separator',
                                  options=[',', ';', ' ', '\t', ':'],
                                  )
                st.session_state.args['sep'] = sep
            else:
                st.session_state.args['sep'] = ',' 

        with prec_check.container():
            mark_precision = st.checkbox('Precision', key='mark_precision')
            if mark_precision:
                prec_no = st.number_input('No digits:',
                                          min_value=0,
                                          max_value=10,
                                          step=1,
                                          value=4)
                st.session_state.args['precision'] = int(prec_no)
            else:
                st.session_state.args['precision'] = 4 

        with formula_check.container():
            mark_formula = st.checkbox('Formula')
            if mark_formula:
                formula_text = st.text_input('Formula')
                st.session_state.args['formula'] = formula_text
            else:
                st.session_state.args['formula'] = None

        with verbose_col.container():
            verbose_mark = st.checkbox('Verbose')
            if verbose_mark:
                st.session_state.args['verbose'] = True
            else:
                st.session_state.args['verbose'] = False

        # === synchronizacja args po każdej zmianie ===
        update_args()


        # === podsumowanie argumentów - wyświetla info ===
        st.markdown("---  \n# Args for script")
        
        show_args = st.checkbox("Show", value=False, key="show_args") 
        if show_args:
            msg = "### Args:  \n"
            for name, val in st.session_state.args.items():
                msg += f"  - `{name}`:&emsp;{val}  \n"
            st.markdown(msg)

    # ================= uruchamianie skryptu =====================
    run = st.button('Run script', key='run_script')
    if run:
        if st.session_state.pth1:
            args = SimpleNamespace(**st.session_state.args)
            # args = afn.args_validation(args, **{"script_name": __file__})
            writer = _WriteToStreamlit(st.empty())

            # with contextlib.redirect_stdout(writer):
            with (contextlib.redirect_stdout(writer),
                  contextlib.redirect_stderr(writer)):
                try:
                    args = afn.args_validation(args, **{"script_name": __file__})
                    main(args)
                except SystemExit as e:
                    # jeśli w innym miejscu użyjesz sys.exit("komunikat"),
                    # to trafi tu jako e.code (string). W Twojej obecnej ścieżce (exit(1))
                    # ten blok po prostu łagodnie kończy przebieg.
                    if isinstance(e.code, str) and e.code.strip():
                        writer.write(e.code + "\n")
                    st.stop() 

            # with contextlib.redirect_stdout(writer):
            #     main(args)
        else:
            st.info('Select arguments, please!')

