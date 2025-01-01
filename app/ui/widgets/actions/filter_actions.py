from typing import TYPE_CHECKING

from PySide6 import QtWidgets

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

def filter_target_videos(main_window: 'MainWindow', search_text: str = ''):
    main_window.target_videos_filter_worker.stop_thread()
    main_window.target_videos_filter_worker.search_text = search_text
    main_window.target_videos_filter_worker.start()

def filter_input_faces(main_window: 'MainWindow', search_text: str = ''):
    main_window.input_faces_filter_worker.stop_thread()
    main_window.input_faces_filter_worker.search_text = search_text
    main_window.input_faces_filter_worker.start()

def filter_merged_embeddings(main_window: 'MainWindow', search_text: str = ''):
    main_window.merged_embeddings_filter_worker.stop_thread()
    main_window.merged_embeddings_filter_worker.search_text = search_text
    main_window.merged_embeddings_filter_worker.start()

def update_filtered_list(main_window: 'MainWindow', filter_list_widget: QtWidgets.QListWidget, visible_indices: list):
    for i in range(filter_list_widget.count()):
        filter_list_widget.item(i).setHidden(True)

    # Show only the items in the visible_indices list
    for i in visible_indices:
        filter_list_widget.item(i).setHidden(False)