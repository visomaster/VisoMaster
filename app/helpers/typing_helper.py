from typing import Dict, Callable

LayoutDictTypes = Dict[str, Dict[str, Dict[str, int|str|list|float|bool|Callable]]]

ParametersTypes = Dict[int, Dict[str, bool|int|float|str]]

ControlTypes = Dict[str, bool|int|float|str]