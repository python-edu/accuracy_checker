# import streamlit as st
import os
import re
from pathlib import Path


class StreamlitFileBrowser:
    exclude_dirs = ["AppData", ".local", ".bin"]
    def __init__(self,
                 root: str | Path = Path.home(),
                 types={'csv', '*.csv'},
                 exclude: list[str] = None
                 ):
        self.root = Path(root).resolve()
        self.type = types.copy()
        if exclude is not None:
            if not isinstance(exclude, list):
                exclude = [exclude]
            self.exclude_dirs.extend(exclude)

    def __call__(self, dir_path: str | Path = None):
        dir_path = Path(dir_path) if dir_path is not None else self.root
        # list_dirs = self._directory_list(dir_path)


class ListDirectories:
    EXCLUDE_DIRS = ["AppData", ".local", ".bin", "__pycache__",
                    ".ipynb_checkpoints", ".env", "env", "venv", ".jupyter",
                    ".git", ".pytest_cache", ".deps",
                    "build", "share", "var", "lib", "etc", "perl",".egg_info",
                    "vim", "neovim", "git", "snap"]

    # pattern dosłowny bez znaków regex np ^$+ itd
    EXCLUDE_LITERALS = ['perl',"conda", "miniconda"]

    # pattern ze znakami regex
    EXCLUDE_REGEX = [r'^[._]',r'^env', r'.env', r'env_.+', r'^venv']

    def __init__(self,
                 root: str | Path = Path.home(),
                 exclude: list[str] = None,
                 ):
        self.root = Path(root).resolve()

        self.exclude_dirs = type(self).EXCLUDE_DIRS[:]
        if exclude is not None:
            if not isinstance(exclude, list):
                exclude = [exclude]
            self.exclude_dirs.extend(exclude)
        
        self.exclude_literals = type(self).EXCLUDE_LITERALS[:]
        self.exclude_regex = type(self).EXCLUDE_REGEX[:]
        self.pattern = self._make_pattern(self.exclude_literals,
                                          self.exclude_regex)
        self.__call__()
        
    def __call__(self, root: str | Path = None):
        root = root or self.root
        self.root = Path(root).resolve()
        self.list_dirs = self._directory_list(self.root)

    def _make_pattern(self, literals: list[str], regex: list[str]):
        """Tworzy patterna dla re ze listy słów."""
        literals = list(map(re.escape, literals))
        regex.extend(literals)
        pattern = '|'.join(regex)
        pattern = re.compile(pattern)
        return pattern

    def _check_exclude_dirs(self, dir_name: str) -> bool:
        """Zwraca True jeśli katalog jest na liście wykluczonych."""
        return dir_name in self.exclude_dirs

    def _check_exclude_regex(self, dir_name: str) -> bool:
        return bool(self.pattern.search(dir_name))

    def _directory_list(self, dir_path: Path):
        res: list[str] = []
        with os.scandir(dir_path) as current:
            for it in current:
                if it.is_dir():
                    if self._check_exclude_dirs(it.name):
                        continue

                    if self._check_exclude_regex(it.name):
                        continue
                    
                    pth = Path(it.path).resolve()
                    res.append(str(pth))
        
        if res is None:
            return []

        return sorted(res)


