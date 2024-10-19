import App.UI.Widgets.WidgetActions as widget_actions
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow
import torch
import qdarkstyle
from PySide6 import QtWidgets 

'''
    Define functions here that has to be executed when value of a control widget (In the settings tab) is changed.
    The first two parameters should be the MainWindow object and the new value of the control 
'''

def change_execution_provider(main_window: 'MainWindow', new_provider):
    main_window.video_processor.stop_processing()
    main_window.models_processor.switch_providers_priority(new_provider)
    main_window.models_processor.delete_models()
    torch.cuda.empty_cache()

def change_threads_number(main_window: 'MainWindow', new_threads_number):
    main_window.video_processor.set_number_of_threads(new_threads_number)
    torch.cuda.empty_cache()

def change_theme(main_window: 'MainWindow', new_theme):
    app = QtWidgets.QApplication.instance()

    if new_theme == "Default":
        with open("App/UI/Styles/styles.qss", "r") as f:
            _style = f.read()
            app.setStyleSheet(_style)  # Applica lo stile predefinito
    elif new_theme == "Dark":  # Correzione: usa "==" invece di "="
        app.setStyleSheet(qdarkstyle.load_stylesheet())  # Applica lo stile dark
    elif new_theme == "Custom":
        with open("App/UI/Styles/custom_theme.qss", "r") as f:
            _style = f.read()
            app.setStyleSheet(_style)  # Applica lo stile custom

    main_window.update()  # Aggiorna la finestra principale