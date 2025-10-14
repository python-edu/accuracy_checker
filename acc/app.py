from pathlib import Path
from types import SimpleNamespace

import streamlit as st
from streamlit_navigation_bar import st_navbar
import numpy as np
import pandas as pd

# import tkinter as tk
# from tkinter import filedialog

# local imports
from acc.src.args_data import help_info
from acc.src.args_data import streamlit_args

# setup for help
help_map = {"Usage": "info_usage",
            "About data": "info_data",
            "About metrics": "info_metrics",
            "Formula": "info_formula"}

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


# === funkcje używane w `Calculations` ===
def get_name(dir_path: str):
    name = Path(dir_path).name
    return name

def setup_dirs():
    if "root" not in st.session_state:
        root = Path(__file__).parent.parent
        root = root.resolve()
        st.session_state['root'] = root

    if "cwd" not in st.session_state:
        cwd = (Path(st.session_state.root) / "example").resolve()
        st.session_state["cwd"] = cwd

    if 'fdir' not in st.session_state:
        cwd = st.session_state.get("cwd", "")
        fdir = streamlit_args.ListDirectories(root=cwd)
        st.session_state['fdir'] = fdir


def reset_dirs():
        root = Path(__file__).parent.parent
        root = root.resolve()
        st.session_state['root'] = root

        cwd = (Path(st.session_state.root) / "example").resolve()
        st.session_state["cwd"] = cwd

        st.session_state.fdir(cwd)


def format_msg(level=3):
    paths = []
    for pth in ('pth1', 'pth2', 'pth3'):
        pth = st.session_state[pth]
        if pth is not None:
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
            paths.append('---')

    return paths


# === start program ===

# to wyświetla pomoc
if page != "Calculations":
    # --- help section
    st.write(f"## {page}")
    text_name = help_map.get(page, 'no')
    help_text = getattr(help_info, text_name, "No help")
    help_text = help_info.parse_help_text(help_text)
    st.markdown(help_text)


# to uruchamia obliczenia - skrypt `main.py`
if page == "Calculations":
    file_pattern = {'File_1': ['*.csv', '*.tif', '*.tiff', '*.TIF', '*.TIFF'],
                    'File_2': ['*.csv', '*.json'],
                    'File_3': ['*.csv']
                    }
    file_name = {'File_1': 'pth1',
                 'File_2': 'pth2',
                 'File_3': 'pth3' 
                    }
    # setup
    setup_dirs()

    for pth in ('pth1', 'pth2', 'pth3'):
        st.session_state.setdefault(pth, None)

    st.session_state.setdefault('choice', 'File_1')
    st.session_state.setdefault('args', SimpleNamespace())

    #with st.sidebar.form("param_form"):
    with st.sidebar:
        options=["File_1", "File_2", "File_3"]
        index = options.index(st.session_state.choice)

        choice = st.radio(
                "Select positional argument:",
            options=options,
            index=index,
            horizontal=True,            # w jednej linii 
        )

        if choice != st.session_state.choice:
            st.session_state.choice = choice

        choice = st.session_state.choice
        cwd = st.session_state.cwd
        st.session_state.fdir(cwd)
        list_dirs = st.session_state.fdir.list_dirs
        
        # info about folder cwd
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

        with c_dir.container():
            new_cwd = st.selectbox("Data dir:",
                                   options=list_dirs,
                                   format_func=get_name,
                                   index=None,
                                   placeholder="— Select folder —",
                                   key='select_dir')

            if new_cwd is not None:
                new_cwd = Path(new_cwd).resolve()
                if new_cwd != cwd:
                    st.session_state["cwd"] = new_cwd
                    st.rerun()

        with c_file.container():
            patterns = file_pattern[choice]
            file_list = [pth for pat in patterns for pth in cwd.glob(pat)]
            file_list = sorted(file_list)
            new_pth = st.selectbox('Select file no 1:',
                                   options = file_list,
                                   format_func=get_name,
                                   index=None,
                                   placeholder="— Select file —",
                                   key=f'select_file_{choice}')

            name = file_name[choice]  # -> pth1, pth2, pth3
            # pth = st.session_state.get(name)
            #name = '---' if pth1 is None else pth1.name 
            #st.markdown(f"> `file1`: **{name}**")
            if new_pth is not None:
                new_pth = Path(new_pth).resolve()
                if new_pth != st.session_state[name]:
                    st.session_state[name] = new_pth
                    st.rerun()
            
        paths_repr = format_msg()
        st.markdown(f">  - `file1`:&emsp;**{paths_repr[0]}**  \n"
                    f">  - `file2`:&emsp;**{paths_repr[1]}**  \n"
                    f">  - `file2`:&emsp;**{paths_repr[2]}**  \n")


        with c_up.container():
            # on_off: kiedy cwd == home to wyłącza przycisk, żeby nie było
            # próby wychodzenia poza katalog domowy i eliminuje błąd
            on_off = cwd == Path.home() 
            if st.button("\u2191", disabled=on_off, width="stretch", key='up_butt'):
                st.session_state["cwd"] = cwd.parent.resolve()
                st.rerun()
        
        with c_home.container():
            if st.button("~", disabled=False, width="stretch", key='home_butt'):
                st.session_state["cwd"] = Path.home()
                st.rerun()
