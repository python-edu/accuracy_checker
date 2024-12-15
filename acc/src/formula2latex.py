import re
from pytexit import py2tex


class LatexFormula:
    def __init__(self, formula):
        self.raw_formula = formula.strip().upper()
        self.metrics, formula = self._split_formula()
        # breakpoint()
        
        formula = py2tex(formula, print_formula=False, print_latex=False) 
        formula = self._add_cdot(formula)
        self.formula = formula

    def _split_formula(self):
        splited = [it.strip() for it in self.raw_formula.split('=')]
        if len(splited) == 2:
            metrics, formula = splited
        else:
            formula = splited[0]
            metrics = 'custom'
        return metrics, formula

    def _add_cdot(self, formula):
        formula = re.sub(r' ', r' \\cdot ', formula)
        return formula

    def __repr__(self):
        if hasattr(self, 'formula'):
            return f"\n\t{self.formula}\n"
        else:
            return f"\n\t{self.raw_formula}\n"

    @property
    def formula(self):
        formula = self._formula[2:]
        return rf"<div>$${self.metrics} = {formula}</div>"

    @formula.setter
    def formula(self, value):
        self._formula = value
