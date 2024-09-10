from PySide6.QtWidgets import QPushButton

class ToggleButton(QPushButton):
    def __init__(self, media_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.media_path = media_path
        self.setCheckable(True)
        self.setStyleSheet("background-color: yellow;")  # Default color for unselected state
        # self.clicked.connect(self.toggle_state)

    def toggle_state(self):
        print(self.media_path)
        print(self.isChecked())
        if self.isChecked():
            self.setStyleSheet("""
                QPushButton {
                    background-color: lightblue;
                    border: 2px solid blue;
                    border-radius: 5px;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: lightgray;
                    border: 2px solid gray;
                    border-radius: 5px;
                }
            """)