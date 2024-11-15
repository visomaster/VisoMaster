from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow

import copy
import App.UI.Widgets.Actions.CommonActions as common_widget_actions
from App.UI.Widgets.Actions import GraphicsViewActions as graphics_view_actions
import cv2
import numpy
import os
from PIL import Image
from PySide6 import QtGui,QtWidgets,QtCore

def set_up_video_seek_line_edit(main_window: 'MainWindow'):
    video_processor = main_window.video_processor
    videoSeekLineEdit = main_window.videoSeekLineEdit
    videoSeekLineEdit.setAlignment(QtCore.Qt.AlignCenter)
    videoSeekLineEdit.setText('0')
    videoSeekLineEdit.setValidator(QtGui.QIntValidator(0, video_processor.max_frame_number))  # Restrict input to numbers 

def set_up_video_seek_slider(main_window: 'MainWindow'):
    main_window.videoSeekSlider.markers = set()  # Store unique tick positions
    main_window.videoSeekSlider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)  # Default position for tick marks

    def addMarker(self, value=None):
        """Add a tick mark at a specific slider value."""
        if value is None or isinstance(value, bool):  # Default to current slider value
            value = self.value()
        if self.minimum() <= value <= self.maximum() and value not in self.markers:
            self.markers.add(value)
            self.update()

    def removeMarker(self, value=None):
        """Remove a tick mark."""
        if value is None or isinstance(value, bool):  # Default to current slider value
            value = self.value()
        if value in self.markers:
            self.markers.remove(value)
            self.update()

    def paintEvent(self: QtWidgets.QSlider, event):
        # Dont need a seek slider if the current selected file is an image
        if main_window.video_processor.file_type=='image':
            return super(QtWidgets.QSlider, self).paintEvent(event)
        # Set up the painter and style option
        painter = QtWidgets.QStylePainter(self)
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        style = self.style()

        # Get groove and handle geometry
        groove_rect = style.subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider, opt, QtWidgets.QStyle.SubControl.SC_SliderGroove
        )
        groove_y = (groove_rect.top() + groove_rect.bottom()) // 2  # Groove's vertical center
        groove_start = groove_rect.left()
        groove_end = groove_rect.right()
        groove_width = groove_end - groove_start

        # Calculate handle position based on the current slider value
        normalized_value = (self.value() - self.minimum()) / (self.maximum() - self.minimum())
        handle_center_x = groove_start + normalized_value * groove_width

        # Make the handle thinner
        handle_width = 5  # Fixed width for thin handle
        handle_height = groove_rect.height()  # Slightly shorter than groove height
        handle_left_x = handle_center_x - (handle_width // 2)
        handle_top_y = groove_y - (handle_height // 2)

        # Define the handle rectangle
        handle_rect = QtCore.QRect(
            handle_left_x, handle_top_y, handle_width, handle_height
        )

        # Draw the groove
        painter.setPen(QtGui.QPen(QtGui.QColor("gray"), 3))  # Groove color and thickness
        painter.drawLine(groove_start, groove_y, groove_end, groove_y)

        # Draw the thin handle
        painter.setPen(QtGui.QPen(QtGui.QColor("white"), 1))  # Handle border color
        painter.setBrush(QtGui.QBrush(QtGui.QColor("white")))  # Handle fill color
        painter.drawRect(handle_rect)

        # Draw markers (if any)
        if self.markers:
            painter.setPen(QtGui.QPen(QtGui.QColor("#e8483c"), 2))  # Marker color and thickness
            for value in sorted(self.markers):
                # Calculate marker position
                marker_normalized_value = (value - self.minimum()) / (self.maximum() - self.minimum())
                marker_x = groove_start + marker_normalized_value * groove_width
                painter.drawLine(marker_x, groove_rect.top(), marker_x, groove_rect.bottom())

    main_window.videoSeekSlider.addMarker = addMarker.__get__(main_window.videoSeekSlider)
    main_window.videoSeekSlider.removeMarker = removeMarker.__get__(main_window.videoSeekSlider)
    main_window.videoSeekSlider.paintEvent = paintEvent.__get__(main_window.videoSeekSlider)

def add_video_slider_marker(main_window: 'MainWindow'):
    current_position = int(main_window.videoSeekSlider.value())
    print("current_position", current_position)
    if not main_window.target_faces:
        common_widget_actions.create_and_show_messagebox(main_window, 'No Target Face Found', 'You need to have atleast one target face to create a marker', main_window.videoSeekSlider)
    elif main_window.markers.get(current_position):
        common_widget_actions.create_and_show_messagebox(main_window, 'Marker Already Exists!', 'A Marker already exists for this position!', main_window.videoSeekSlider)
    else:
        main_window.videoSeekSlider.addMarker(current_position)
        main_window.markers[current_position] = copy.deepcopy(main_window.parameters)
        print("Adding marker: ", main_window.parameters.copy())
        print(f"Marker Added for Frame: {current_position}")

def remove_video_slider_marker(main_window: 'MainWindow'):
    current_position = int(main_window.videoSeekSlider.value())
    print("current_position", current_position)
    if main_window.markers.get(current_position):
        main_window.videoSeekSlider.removeMarker(current_position)
        main_window.markers.pop(current_position)
        print(f"Marker Removed from position: {current_position}")
    else:
        common_widget_actions.create_and_show_messagebox(main_window, 'No Marker Found!', 'No Marker Found for this position!', main_window.videoSeekSlider)

def move_slider_to_nearest_marker(main_window: 'MainWindow', direction: str):
    """
    Move the slider to the nearest marker in the specified direction.

    :param direction: 'next' to move to the next marker, 'previous' to move to the previous marker.
    """
    current_position = int(main_window.videoSeekSlider.value())
    markers = sorted(main_window.markers.keys())
    if direction == "next":
        filtered_markers = [marker for marker in markers if marker > current_position]
        new_position = filtered_markers[0] if filtered_markers else None
    elif direction == "previous":
        filtered_markers = [marker for marker in markers if marker < current_position]
        new_position = filtered_markers[-1] if filtered_markers else None

    if new_position is not None:
        main_window.videoSeekSlider.setValue(new_position)
        main_window.video_processor.process_current_frame()

# Wrappers for specific directions
def move_slider_to_next_nearest_marker(main_window: 'MainWindow'):
    move_slider_to_nearest_marker(main_window, "next")

def move_slider_to_previous_nearest_marker(main_window: 'MainWindow'):
    move_slider_to_nearest_marker(main_window, "previous")

def remove_face_parameters_from_markers(main_window: 'MainWindow', face_id):
    for frame_number, parameters in main_window.markers.items():
        parameters.pop(face_id)
        # If the parameters is empty, then there is no longer any marker to be set for any target face
        if not parameters:
            delete_all_markers(main_window)
            break

def delete_all_markers(main_window: 'MainWindow'):
    main_window.videoSeekSlider.markers = set()
    main_window.videoSeekSlider.update()
    main_window.markers = {}

def view_fullscreen(main_window: 'MainWindow'):
    if main_window.is_full_screen:
        main_window.showNormal()  # Exit full-screen mode
    else:
        main_window.showFullScreen()  # Enter full-screen mode
    main_window.is_full_screen = not main_window.is_full_screen

def enable_zoom_and_pan(view: QtWidgets.QGraphicsView):
    SCALE_FACTOR = 1.5
    view._zoom = 0  # Track zoom level
    view._last_scale_factor = 1.0  # Track the last scale factor (1.0 = no scaling)

    def zoom(self, step=False):
        """Zoom in or out by a step."""
        if not step:
            factor = self._last_scale_factor
        else:
            self._zoom += step
            factor = SCALE_FACTOR ** step
            self._last_scale_factor *= factor  # Update the last scale factor
        if factor > 0:
            self.scale(factor, factor)

    def wheelEvent(self, event):
        """Handle mouse wheel event for zooming."""
        delta = event.angleDelta().y()
        if delta != 0:
            zoom(self, delta // abs(delta))
    
    def resetZoom(self):
        print("Called resetZoom()")
        """Reset zoom level to fit the view."""
        self._zoom = 0
        if not self.scene():
            return
        items = self.scene().items()
        if not items:
            return
        rect = self.scene().itemsBoundingRect()
        self.setSceneRect(rect)
        unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
        self.scale(1 / unity.width(), 1 / unity.height())
        view_rect = self.viewport().rect()
        scene_rect = self.transform().mapRect(rect)
        factor = min(view_rect.width() / scene_rect.width(),
                    view_rect.height() / scene_rect.height())
        self.scale(factor, factor)

    # Attach methods to the view
    view.zoom = zoom.__get__(view)
    view.resetZoom = resetZoom.__get__(view)
    view.wheelEvent = wheelEvent.__get__(view)

    # Set anchors for better interaction
    view.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
    view.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)


def OnClickPlayButton(main_window: 'MainWindow', checked):
    video_processor = main_window.video_processor
    if checked:
        if video_processor.processing or video_processor.current_frame_number==video_processor.max_frame_number:
            print("OnClickPlayButton: Video already playing. Stopping the current video before starting a new one.")
            video_processor.stop_processing()
            return
        print("OnClickPlayButton: Starting video processing.")
        setPlayButtonIconToStop(main_window)
        video_processor.process_video()
    else:
        video_processor = main_window.video_processor
        print("OnClickPlayButton: Stopping video processing.")
        setPlayButtonIconToPlay(main_window)
        video_processor.stop_processing()
        main_window.buttonMediaRecord.blockSignals(True)
        main_window.buttonMediaRecord.setChecked(False)
        main_window.buttonMediaRecord.blockSignals(False)
        setRecordButtonIconToPlay(main_window)


def OnClickRecordButton(main_window: 'MainWindow', checked):
    video_processor = main_window.video_processor
    if checked:
        if video_processor.processing or video_processor.current_frame_number==video_processor.max_frame_number:
            print("OnClickRecordButton: Video already playing. Stopping the current video before starting a new one.")
            video_processor.stop_processing()
            return
        video_processor.recording = True
        main_window.buttonMediaPlay.setChecked(True)
        setRecordButtonIconToStop(main_window)

    else:
        main_window.buttonMediaPlay.setChecked(False)
        video_processor.stop_processing()
        setPlayButtonIconToPlay(main_window)
        setRecordButtonIconToPlay(main_window)

def setRecordButtonIconToPlay(main_window: 'MainWindow'):
    main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/Media/rec_off.png"))
    main_window.buttonMediaRecord.setToolTip("Start Recording")
def setRecordButtonIconToStop(main_window: 'MainWindow'):
    main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/Media/rec_on.png"))
    main_window.buttonMediaRecord.setToolTip("Stop Recording")

def setPlayButtonIconToPlay(main_window: 'MainWindow'):
    main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/Media/play_off.png"))
    main_window.buttonMediaPlay.setToolTip("Play")

def setPlayButtonIconToStop(main_window: 'MainWindow'):
    main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/Media/play_on.png"))
    main_window.buttonMediaPlay.setToolTip("Stop")

def resetMediaButtons(main_window: 'MainWindow'):
    # Rest the state and icons of the buttons without triggering Onchange methods
    main_window.buttonMediaPlay.blockSignals(True)
    main_window.buttonMediaPlay.setChecked(False)
    main_window.buttonMediaPlay.blockSignals(False)
    main_window.buttonMediaRecord.blockSignals(True)
    main_window.buttonMediaRecord.setChecked(False)
    main_window.buttonMediaRecord.blockSignals(False)
    setPlayButtonIcon(main_window)
    setRecordButtonIcon(main_window)


def setPlayButtonIcon(main_window: 'MainWindow'):
    if main_window.buttonMediaPlay.isChecked(): 
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/Media/play_on.png"))
        main_window.buttonMediaPlay.setToolTip("Stop")
    else:
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/Media/play_off.png"))
        main_window.buttonMediaPlay.setToolTip("Play")

def setRecordButtonIcon(main_window: 'MainWindow'):
    if main_window.buttonMediaRecord.isChecked(): 
        main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/Media/rec_on.png"))
        main_window.buttonMediaRecord.setToolTip("Stop Recording")
    else:
        main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/Media/rec_off.png"))
        main_window.buttonMediaRecord.setToolTip("Start Recording")

@QtCore.Slot(int)
def OnChangeSlider(main_window: 'MainWindow', new_position=0):
    print("Called OnChangeSlider()")
    video_processor = main_window.video_processor

    was_processing = video_processor.stop_processing()
    if was_processing:
        print("OnChangeSlider: Processing in progress. Stopping current processing.")

    video_processor.current_frame_number = new_position
    video_processor.next_frame_to_display = new_position
    if video_processor.media_capture:
        video_processor.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_position)
        ret, frame = video_processor.media_capture.read()
        if ret:
            pixmap = common_widget_actions.get_pixmap_from_frame(main_window, frame)
            graphics_view_actions.update_graphics_view(main_window, pixmap, new_position)
            # restore slider position 
            video_processor.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_position)
            update_parameters_from_marker(main_window, new_position)
            update_widget_values_from_markers(main_window, new_position)

    # Do not automatically restart the video, let the user press Play to resume
    print("OnChangeSlider: Video stopped after slider movement.")

def update_parameters_from_marker(main_window: 'MainWindow', new_position):
    if main_window.markers.get(new_position):
        main_window.parameters = copy.deepcopy(main_window.markers[new_position])

def update_widget_values_from_markers(main_window: 'MainWindow', new_position):
    if main_window.markers.get(new_position):
        if main_window.selected_target_face_id is not None:
            common_widget_actions.set_widgets_values_using_face_id_parameters(main_window, main_window.selected_target_face_id)

def on_slider_moved(main_window: 'MainWindow'):
    # print("Called on_slider_moved()")
    position = main_window.videoSeekSlider.value()
    print(f"\nSlider Moved. position: {position}\n")

def on_slider_pressed(main_window: 'MainWindow'):

    position = main_window.videoSeekSlider.value()
    print(f"\nSlider Pressed. position: {position}\n")

def on_slider_released(main_window: 'MainWindow'):
    # print("Called on_slider_released()")

    new_position = main_window.videoSeekSlider.value()
    print(f"\nSlider released. New position: {new_position}\n")
    # Perform the update to the new frame
    video_processor = main_window.video_processor
    if video_processor.media_capture:
        video_processor.process_current_frame()  # Process the current frame

def process_swap_faces(main_window: 'MainWindow'):
    video_processor = main_window.video_processor
    video_processor.process_current_frame()

def process_edit_faces(main_window: 'MainWindow'):
    video_processor = main_window.video_processor
    video_processor.process_current_frame()


def save_current_frame_to_file(main_window: 'MainWindow'):
    frame = main_window.video_processor.current_frame.copy()
    if isinstance(frame, numpy.ndarray):
        save_filename, extension = os.path.splitext(main_window.video_processor.media_path)
        save_filename, _ = QtWidgets.QFileDialog.getSaveFileName(main_window, 'Save Frame as Image', f'{save_filename}.png', filter='PNG (*.png)',)
        if save_filename:
            pil_image = Image.fromarray(frame[..., ::-1])
            pil_image.save(save_filename, 'PNG')
    else:
        common_widget_actions.create_and_show_messagebox(main_window, 'Invalid Frame', 'Cannot save the current frame!', parent_widget=main_window.saveImageButton)
