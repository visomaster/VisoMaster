from PySide6 import QtWidgets
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow
from App.UI.Widgets.WidgetComponents import *
from App.UI.Widgets.SettingsLayoutData import SETTINGS_LAYOUT_DATA
from pyqttoast import Toast, ToastPreset, ToastPosition
import threading
import numpy
import json

def create_and_show_messagebox(main_window: 'MainWindow', window_title: str, message: str, parent_widget: QtWidgets.QWidget):
    messagebox = QtWidgets.QMessageBox(parent_widget)
    messagebox.setWindowTitle(window_title)
    messagebox.setText(message)
    messagebox.exec_()

def create_and_show_toast_message(main_window: 'MainWindow', title: str, message: str, style_type='information'):
    style_preset_map = {
        'success': ToastPreset.SUCCESS,
        'warning': ToastPreset.WARNING,
        'error': ToastPreset.ERROR,
        'information': ToastPreset.INFORMATION,
        'success_dark': ToastPreset.SUCCESS_DARK,
        'warning_dark': ToastPreset.WARNING_DARK,
        'error_dark': ToastPreset.ERROR_DARK,
        'information_dark': ToastPreset.INFORMATION_DARK,
    }
    toast = Toast(main_window)
    toast.setTitle(title)
    toast.setText(message)
    toast.setDuration(1400)
    toast.setPosition(ToastPosition.TOP_RIGHT)  # Default: ToastPosition.BOTTOM_RIGHT
    toast.applyPreset(style_preset_map[style_type])  # Apply style preset
    toast.show()
    
def create_control(main_window: 'MainWindow', control_name, control_value):
    main_window.control[control_name] = control_value

def update_control(main_window: 'MainWindow', control_name, control_value, exec_function=None, exec_function_args=[]):
    if exec_function:
        # Only execute the function if the value is different from current
        if main_window.control[control_name] != control_value:
            # By default an exec function definition should have atleast one parameter : MainWindow
            exec_function_args = [main_window, control_value] + exec_function_args
            exec_function(*exec_function_args)
    main_window.control[control_name] = control_value
    refresh_frame(main_window)

def create_default_parameter(main_window: 'MainWindow', parameter_name, parameter_value):
    main_window.default_parameters[parameter_name] = parameter_value

def create_parameter_dict_for_face_id(main_window: 'MainWindow', face_id=0):
    if not main_window.parameters.get(face_id):
        parameters =  main_window.parameters.get(main_window.selected_target_face_id) or main_window.default_parameters
        main_window.parameters[face_id] = parameters.copy()
    print("Created parameter_dict_for_face_id", face_id)

def update_parameter(main_window: 'MainWindow', parameter_name, parameter_value, enable_refresh_frame=True):
    current_position = main_window.videoSeekSlider.value()
    face_id = main_window.selected_target_face_id

    # Update marker parameters too
    if main_window.markers.get(current_position) and face_id:
        main_window.markers[current_position][face_id][parameter_name] = parameter_value

    if main_window.target_faces:
        main_window.parameters[face_id][parameter_name] = parameter_value

        if enable_refresh_frame:
            refresh_frame(main_window)

def refresh_frame(main_window: 'MainWindow'):
    video_processor = main_window.video_processor
    video_processor.process_current_frame()

# Function to Hide Elements conditionally from values in LayoutData (Currently supports using Selection box and Toggle button to hide other widgets)
def show_hide_related_widgets(main_window: 'MainWindow', parent_widget, parent_widget_name: str, value1=False, value2=False):
    if main_window.parameter_widgets:
        group_layout_data = parent_widget.group_layout_data #Dictionary contaning layout data of all elements in the group of the parent_widget
        if 'Selection' in parent_widget_name:
            # Loop through all widgets data in the parent widget's group layout data
            for widget_name in group_layout_data.keys():
                # Store the widget object (instance) from the parameters_widgets Dictionary
                current_widget = main_window.parameter_widgets.get(widget_name, False)
                # Check if the current_widget depends on the Parent Widget's (selection) value 
                if group_layout_data[widget_name].get('parentSelection', '') == parent_widget_name and current_widget:
                    # Check if the current_widget has the required value of Parent Widget's (selection) current value to hide/show the current_widget
                    if group_layout_data[widget_name].get('requiredSelectionValue') != parent_widget.currentText():
                        current_widget.hide()
                        current_widget.label_widget.hide()
                        current_widget.reset_default_button.hide()
                        if current_widget.line_edit:
                            current_widget.line_edit.hide()
                    else:
                        current_widget.show()
                        current_widget.label_widget.show()
                        current_widget.reset_default_button.show()
                        if current_widget.line_edit:
                            current_widget.line_edit.show()

        elif 'Toggle' in parent_widget_name:
            # Loop through all widgets data in the parent widget's group layout data
            for widget_name in group_layout_data.keys():
                # Store the widget object (instance) from the parameters_widgets Dictionary
                if not widget_name in main_window.parameter_widgets:
                    continue
                current_widget = main_window.parameter_widgets[widget_name]
                # Check if the current_widget depends on the Parent Widget's (toggle) value 
                parentToggles = group_layout_data[widget_name].get('parentToggle', '')
                if parent_widget_name in parentToggles:
                    if ',' in parentToggles:
                        result = [item.strip() for item in parentToggles.split(',')]
                        parentToggle_ischecked = False
                        for index, required_widget_name in enumerate(result):
                            parentToggle_ischecked = main_window.parameter_widgets[required_widget_name].isChecked()
                        # Check if the current_widget has the required toggle value of Parent Widget's (toggle) checked state to hide/show the current_widget
                        if group_layout_data[widget_name].get('requiredToggleValue') != parentToggle_ischecked:
                            current_widget.hide()
                            current_widget.label_widget.hide()
                            current_widget.reset_default_button.hide()
                            if current_widget.line_edit:
                                current_widget.line_edit.hide()
                        else:
                            current_widget.show()
                            current_widget.label_widget.show()
                            current_widget.reset_default_button.show()
                            if current_widget.line_edit:
                                current_widget.line_edit.show()
                    elif '|' in parentToggles:
                        result = [item.strip() for item in parentToggles.split('|')]
                        parentToggle_ischecked = True
                        for index, required_widget_name in enumerate(result):
                            ischecked = main_window.parameter_widgets[required_widget_name].isChecked()
                            if not ischecked:
                                parentToggle_ischecked = False
                        # Check if the current_widget has the required toggle value of Parent Widget's (toggle) checked state to hide/show the current_widget
                        if group_layout_data[widget_name].get('requiredToggleValue') != parentToggle_ischecked:
                            current_widget.hide()
                            current_widget.label_widget.hide()
                            current_widget.reset_default_button.hide()
                            if current_widget.line_edit:
                                current_widget.line_edit.hide()
                        else:
                            current_widget.show()
                            current_widget.label_widget.show()
                            current_widget.reset_default_button.show()
                            if current_widget.line_edit:
                                current_widget.line_edit.show()
                    else:
                        parentToggle_ischecked = main_window.parameter_widgets[parentToggles].isChecked()
                        if group_layout_data[widget_name].get('requiredToggleValue') != parentToggle_ischecked:
                            current_widget.hide()
                            current_widget.label_widget.hide()
                            current_widget.reset_default_button.hide()
                            if current_widget.line_edit:
                                current_widget.line_edit.hide()
                        else:
                            current_widget.show()
                            current_widget.label_widget.show()
                            current_widget.reset_default_button.show()
                            if current_widget.line_edit:
                                current_widget.line_edit.show()

            parent_widget.start_animation()
            
def get_pixmap_from_frame(main_window: 'MainWindow', frame: np.ndarray):
    height, width, channel = frame.shape
    if channel == 2:
        # Frame in grayscale
        bytes_per_line = width
        q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_Grayscale8)
    else:
        # Frame in color
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888).rgbSwapped()
    pixmap = QtGui.QPixmap.fromImage(q_img)
    return pixmap


def update_gpu_memory_progressbar(main_window: 'MainWindow'):
    threading.Thread(target=partial(_update_gpu_memory_progressbar, main_window)).start()

def _update_gpu_memory_progressbar(main_window: 'MainWindow'):
    memory_used, memory_total = main_window.models_processor.get_gpu_memory()
    main_window.gpu_memory_update_signal.emit(memory_used, memory_total)

@qtc.Slot(int, int)
def set_gpu_memory_progressbar_value(main_window: 'MainWindow', memory_used, memory_total):
    main_window.vramProgressBar.setMaximum(memory_total)
    main_window.vramProgressBar.setValue(memory_used)
    main_window.vramProgressBar.setFormat(f'{round(memory_used/1024,2)} GB / {round(memory_total/1024,2)} GB (%p%)')
    main_window.vramProgressBar.update()

def clear_gpu_memory(main_window: 'MainWindow'):
    main_window.video_processor.stop_processing()
    main_window.models_processor.clear_gpu_memory()
    main_window.swapfacesButton.setChecked(False)
    main_window.editFacesButton.setChecked(False)
    update_gpu_memory_progressbar(main_window)

    main_window.videoSeekSlider.markers = set()
    main_window.videoSeekSlider.update()

def extract_frame_as_pixmap(media_file_path, file_type, webcam_index=False, webcam_backend=False):
    frame = False
    if file_type=='image':
        frame = cv2.imread(media_file_path)
    elif file_type=='video':    
        cap = cv2.VideoCapture(media_file_path)
        ret, frame = cap.read()
        cap.release()
    elif file_type=='webcam':
        camera = cv2.VideoCapture(webcam_index, webcam_backend)
        if not camera.isOpened():
            return
        ret, frame = camera.read()
        if not ret:
            return

    if isinstance(frame, numpy.ndarray):
        # Convert the frame to QPixmap
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(70, 70, qtc.Qt.AspectRatioMode.KeepAspectRatio)  # Adjust size as needed
        return pixmap
    return None

def set_widgets_values_using_face_id_parameters(main_window: 'MainWindow', face_id=False):
    if face_id is False:
        print("Set widgets values using default parameters")
        parameters = main_window.default_parameters
    else:
        print(f"Set widgets values using face_id {face_id}")
        parameters = main_window.parameters[face_id].copy()
    parameter_widgets = main_window.parameter_widgets
    for parameter_name, parameter_value in parameters.items():
        # temporarily disable refreshing the frame to prevent slowing due to unnecessary processing
        parameter_widgets[parameter_name].enable_refresh_frame = False
        parameter_widgets[parameter_name].set_value(parameter_value)
        parameter_widgets[parameter_name].enable_refresh_frame = True

def set_control_widgets_values(main_window: 'MainWindow'):
    """
    Set the values of control widgets based on the `control` data in the `main_window`.

    Temporarily disables frame refreshing while setting values to avoid unnecessary processing.
    """
    # Get control values and parameter widgets from the main window
    control = main_window.control.copy()
    parameter_widgets = main_window.parameter_widgets

    # Prepare a dictionary of settings options from layout data
    settings_options = {
        setting_name: setting_data
        for setting_group in SETTINGS_LAYOUT_DATA.values()
        for setting_name, setting_data in setting_group.items()
    }

    # Iterate through control items and update widgets
    for control_name, control_value in control.items():
        widget = parameter_widgets[control_name]

        # Temporarily disable frame refresh
        widget.enable_refresh_frame = False

        # Set the widget value
        widget.set_value(control_value)

        # Execute any associated function, if defined
        exec_function_data = settings_options[control_name].get('exec_function')
        if exec_function_data:
            exec_function = partial(
                exec_function_data, main_window
            )
            exec_args = settings_options[control_name].get('exec_fuction_args', [])
            exec_function(control_value, *exec_args)

        # Re-enable frame refresh
        widget.enable_refresh_frame = True
        
@qtc.Slot(QtWidgets.QListWidget, bool)
def update_placeholder_visibility(main_window: 'MainWindow', list_widget:QtWidgets.QListWidget, default_hide):
    """Update the visibility of the placeholder text."""
    """
        The default_hide parameter is used to Hide the placeholder text by default. 
        If the default_hide is False, then the visibility of the placeholder text is set using the size of the list_widget 
    """
    if default_hide:
        is_visible = False
    else:
        is_visible = list_widget.count()==0
    list_widget.placeholder_label.setVisible(is_visible)
    # Set Cursor on the List Widget
    if is_visible:
        list_widget.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    else:
        list_widget.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
    print("SetVisible", is_visible)
    print("targetVideosList.count()", list_widget.count())

