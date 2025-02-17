from typing import Dict, Callable, NewType
from app.helpers.miscellaneous import ParametersDict

LayoutDictTypes = NewType('LayoutDictTypes', Dict[str, Dict[str, Dict[str, int|str|list|float|bool|Callable]]])

ParametersTypes = NewType('ParametersTypes', ParametersDict)
FacesParametersTypes = NewType('FacesParametersTypes', dict[int, ParametersTypes])

ControlTypes = NewType('ControlTypes', Dict[str, bool|int|float|str])

MarkerTypes = NewType('MarkerTypes', Dict[int, Dict[str, FacesParametersTypes|ControlTypes]])