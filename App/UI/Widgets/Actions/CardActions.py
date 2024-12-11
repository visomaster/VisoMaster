from PySide6 import QtCore, QtWidgets
from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow
import App.UI.Widgets.Actions.CommonActions as common_widget_actions

def clear_target_faces(main_window: 'MainWindow', refresh_frame=True):
    main_window.targetFacesList.clear()
    for face_id, target_face in main_window.target_faces.items():
        target_face.deleteLater()
    main_window.target_faces = {}
    main_window.parameters = {}

    # Set Parameter widget values to default
    common_widget_actions.set_widgets_values_using_face_id_parameters(main_window=main_window, face_id=False)
    if refresh_frame:
        common_widget_actions.refresh_frame(main_window=main_window)

    
def clear_input_faces(main_window: 'MainWindow'):
    main_window.inputFacesList.clear()
    for face_id, input_face in main_window.input_faces.items():
        input_face.deleteLater()
    main_window.input_faces = {}

    for face_id, target_face in main_window.target_faces.items():
        target_face.assigned_input_face_buttons = {}
        target_face.calculateAssignedInputEmbedding()
    common_widget_actions.refresh_frame(main_window=main_window)

def uncheck_all_input_faces(main_window: 'MainWindow'):
    # Uncheck All other input faces 
    for face_id, input_face_button in main_window.input_faces.items():
        input_face_button.setChecked(False)

def uncheck_all_merged_embeddings(main_window: 'MainWindow'):
    for embedding_id, embed_button in  main_window.merged_embeddings.items():
        embed_button.setChecked(False)