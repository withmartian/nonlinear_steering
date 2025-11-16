from __future__ import annotations

from pathlib import Path
import importlib.util
import sys


DCT_URL = "https://raw.githubusercontent.com/luke-marks0/melbo-dct-post/main/src/dct.py"


def load_dct_module(path: str = "dct.py", url: str = DCT_URL):
    from urllib.request import urlopen
    p = Path(path)
    if not p.exists():
        p.write_text(urlopen(url).read().decode())
    spec = importlib.util.spec_from_file_location("dct", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dct"] = mod
    spec.loader.exec_module(mod)
    return mod



