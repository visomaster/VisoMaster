import App.UI.Widgets.WidgetActions as widget_actions
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow
import torch

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