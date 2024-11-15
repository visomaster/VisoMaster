from PySide6 import QtWidgets
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow

def filterTargetVideos(main_window: 'MainWindow', search_text: str = ''):
    main_window.target_videos_filter_worker.stop_thread()
    main_window.target_videos_filter_worker.search_text = search_text
    main_window.target_videos_filter_worker.start()

def filterInputFaces(main_window: 'MainWindow', search_text: str = ''):
    main_window.input_faces_filter_worker.stop_thread()
    main_window.input_faces_filter_worker.search_text = search_text
    main_window.input_faces_filter_worker.start()

def filterMergedEmbeddings(main_window: 'MainWindow', search_text: str = ''):
    main_window.merged_embeddings_filter_worker.stop_thread()
    main_window.merged_embeddings_filter_worker.search_text = search_text
    main_window.merged_embeddings_filter_worker.start()

def updateFilteredList(main_window: 'MainWindow', filter_list_widget: QtWidgets.QListWidget, visible_indices: list):
    for i in range(filter_list_widget.count()):
        filter_list_widget.item(i).setHidden(True)

    # Show only the items in the visible_indices list
    for i in visible_indices:
        filter_list_widget.item(i).setHidden(False)