from pathlib import Path
from types import SimpleNamespace

import streamlit as st
from streamlit_navigation_bar import st_navbar
import numpy as np
import pandas as pd


from acc.src.args_data import help_info

# setup for help
help_map = {"Usage": "info_usage",
            "About data": "info_data",
            "About metrics": "info_metrics",
            "Formula": "info_formula"}


# setup streamlite
st.set_page_config(page_title="Accuracy", layout="wide")
page = st_navbar(["Calculations", "Usage", "About data", "About metrics",
                  "Formula"], adjust=False)


# setup for global variables
st.session_state.setdefault("DATA_ROOT", )



if page != "Calculations":
    # --- help section
    st.write(f"## {page}")
    text_name = help_map.get(page, 'no')
    help_text = getattr(help_info, text_name, "No help")
    help_text = help_info.parse_help_text(help_text)
    st.markdown(help_text)

if page == "Calculations":
    args = SimpleNamespace()
    with st.sidebar.form("param_form"):
        
        submitted = st.form_submit_button("Save args")


