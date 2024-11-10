import App.UI.Widgets.WidgetActions as widget_actions
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow
import torch
import qdarkstyle
from PySide6 import QtWidgets 
import qdarktheme

'''
    Define functions here that has to be executed when value of a control widget (In the settings tab) is changed.
    The first two parameters should be the MainWindow object and the new value of the control 
'''

def change_execution_provider(main_window: 'MainWindow', new_provider):
    main_window.video_processor.stop_processing()
    main_window.models_processor.switch_providers_priority(new_provider)
    main_window.models_processor.clear_gpu_memory()
    widget_actions.update_gpu_memory_progressbar(main_window)

def change_threads_number(main_window: 'MainWindow', new_threads_number):
    main_window.video_processor.set_number_of_threads(new_threads_number)
    torch.cuda.empty_cache()
    widget_actions.update_gpu_memory_progressbar(main_window)


def change_theme(main_window: 'MainWindow', new_theme):

    def get_style_data(filename, theme='dark', custom_colors={"primary": "#4facc9"}):
        with open(f"App/UI/Styles/{filename}", "r") as f:
            _style = f.read()
            _style = qdarktheme.load_stylesheet(theme=theme, custom_colors=custom_colors)+'\n'+_style
        return _style
    app = QtWidgets.QApplication.instance()

    if new_theme == "Dark":
        _style = get_style_data('dark_styles.qss', 'dark',)

    elif new_theme == "Light":
        _style = get_style_data('light_styles.qss', 'light',)

    elif new_theme == "Dark-Blue":
        _style = qdarkstyle.load_stylesheet() # Applica lo stile dark-blue 

    app.setStyleSheet(_style)

    main_window.update()  # Aggiorna la finestra principale