"""JSON rendering that keeps six-decimal metric strings stable."""

from __future__ import annotations

import json
import re

import numpy as np

_FLOAT_MARKER = re.compile(r'"__FLOAT__([-+0-9.eE]+)__"')


def dumps_rounded(payload) -> str:
    """Pretty JSON. Floats are written with six digits after the decimal point."""

    def convert(value):
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return f"__FLOAT__{float(value):.6f}__"
        return value

    text = json.dumps(convert(payload), indent=2)
    text = _FLOAT_MARKER.sub(r"\1", text)
    return text + "\n"
