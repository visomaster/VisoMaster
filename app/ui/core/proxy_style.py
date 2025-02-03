from PySide6 import QtWidgets
from PySide6.QtCore import Qt


class ProxyStyle(QtWidgets.QProxyStyle):
    def styleHint(self, hint, opt=None, widget=None, returnData=None) -> int:
        res = super().styleHint(hint, opt, widget, returnData)
        if hint == self.StyleHint.SH_Slider_AbsoluteSetButtons:
            res = Qt.LeftButton.value
        return res
