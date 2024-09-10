call conda activate pyside_test
pyside6-uic MainWindow.ui -o MainWindow.py
pyside6-rcc media.qrc -o media_rc.py
pause