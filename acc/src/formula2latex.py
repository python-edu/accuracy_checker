import re


class LatexFormula:
    def __init__(self, formula):
        self.raw_formula = formula.strip()
        self.metrics, formula = self._split_formula()

        formula = self._match_divide(formula)
        formula = self._power_replace(formula)
        formula = self._match_multiplication(formula)
        # breakpoint()
        self.formula = formula

    def _split_formula(self):
        metrics, formula = [it.strip() for it in self.raw_formula.split('=')]
        return metrics, formula

    def _wrap_power_string(self, match) -> str:
        power = match.group(2)
        return f"^{{{ power }}}"

    def _power_replace(self, formula) -> str:
        pattern = r'(\*\*)([\d\.]{1,3})'
        formula = re.sub(pattern, self._wrap_power_string, formula)
        return formula

    def _match_divide(self, formula):
        pattern = r'([-+*^\dTFPN.)(]+)/([-+*^\dTFPN.)(]+)'
        # breakpoint()
        formula = re.sub(pattern, r'\\frac{\1}{\2}', formula)
        return formula

    def _match_multiplication(self, formula):
        pattern = r'[*]'
        formula = re.sub(pattern, r' \\cdot ', formula)
        return formula

    def __repr__(self):
        if hasattr(self, 'formula'):
            return f"\n\t{self.formula}\n"
        else:
            return f"\n\t{self.raw_formula}\n"

    @property
    def formula(self):
        return rf"<div>\[{self.metrics} = {self._formula}\]</div>"

    @formula.setter
    def formula(self, value):
        self._formula = value
