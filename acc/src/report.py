# -*- coding: utf-8 -*-

import datetime
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
# --


"""
The module contains classes providing methods to perform a simple report of
accuracy calculations.
Classes used in scripts of the type 'accuracy.py'
"""
# ---


class AccuracyReport:
    def __init__(self, **kwargs):
        self.params_names = [
            "title",
            "description",
            "report_file",
            "template_file",
            "template_dir",
            "script_name",
            "date",
        ]

        self._set_default()
        if kwargs:
            self._set_args(**kwargs)
            self.env = Environment(loader=FileSystemLoader(self.template_dir))
            self.template = self.env.get_template(self.template_file)

    # ---

    def _set_args(self, **kwargs):
        for name in kwargs.keys():
            if name in self.params_names:
                val = kwargs.get(name)

                if name == "template_dir":
                    val = Path(val).resolve()

                if name == "date":
                    val = datetime.datetime.now().strftime("%Y-%m-%d")

                setattr(self, name, val)

    # --

    def _set_default(self):
        for name in self.params_names:
            val = None
            if name == "date":
                val = datetime.datetime.now().strftime("%Y-%m-%d")

            setattr(self, name, val)

    # ---

    def __repr__(self):
        res = ""
        for name in self.params_names:
            res += f"{name}: {getattr(self, name)}\n"
        return res

    # ---

    def _get_params(self):
        params = {}
        for name in self.params_names:
            params[name] = getattr(self, name)
        return params

    # ---

    def __call__(self, data_dict):
        """
        Args:
            - data_dict: {'title': df, ...}
        """
        table_data = {}
        # breakpoint()
        for title, df in data_dict.items():
            table_data[title] = df.to_html(index=True)

        description_data = self._get_params()
        return self.template.render(
            description_data=description_data, table_data=table_data
        )
        # ---
