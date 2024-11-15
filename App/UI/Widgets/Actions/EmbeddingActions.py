import App.UI.Widgets.Actions.CommonActions as common_widget_actions

from App.UI.Widgets.WidgetComponents import EmbeddingCardButton
from PySide6 import QtWidgets, QtCore
import json
import numpy
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from App.UI.MainUI import MainWindow

def create_and_add_embed_button_to_list(main_window: 'MainWindow', embedding_name, embedding_store):
    inputEmbeddingsList = main_window.inputEmbeddingsList
    # Passa l'intero embedding_store
    embed_button = EmbeddingCardButton(main_window=main_window, embedding_name=embedding_name, embedding_store=embedding_store)

    button_size = QtCore.QSize(120, 30)  # Imposta una dimensione fissa per i pulsanti
    embed_button.setFixedSize(button_size)
    
    list_item = QtWidgets.QListWidgetItem(inputEmbeddingsList)
    list_item.setSizeHint(button_size)
    embed_button.list_item = list_item
    list_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    
    inputEmbeddingsList.setItemWidget(list_item, embed_button)
    
    # Aggiungi padding attorno ai pulsanti
    grid_size_with_padding = button_size + QtCore.QSize(4, 4)
    inputEmbeddingsList.setGridSize(grid_size_with_padding)  # Add padding around the buttons
    inputEmbeddingsList.setWrapping(True)  # Set grid size with padding
    inputEmbeddingsList.setFlow(QtWidgets.QListView.LeftToRight)  # Set flow direction
    inputEmbeddingsList.setResizeMode(QtWidgets.QListView.Adjust)  # Adjust layout automatically

    main_window.merged_embeddings.append(embed_button)

def open_embeddings_from_file(main_window: 'MainWindow'):
    embedding_filename, _ = QtWidgets.QFileDialog.getOpenFileName(main_window, filter='JSON (*.json)')
    if embedding_filename:
        with open(embedding_filename, 'r') as embed_file:
            embeddings_list = json.load(embed_file)
            clear_merged_embeddings(main_window)

            # Reset per ogni target face
            for target_face in main_window.target_faces:
                target_face.assigned_embed_buttons = {}
                target_face.assigned_input_embedding = {}

            # Carica gli embedding dal file e crea il dizionario embedding_store
            for embed_data in embeddings_list:
                embedding_store = embed_data.get('embedding_store', {})
                # Converte ogni embedding in numpy array
                for recogn_model, embed in embedding_store.items():
                    embedding_store[recogn_model] = numpy.array(embed)

                # Passa l'intero embedding_store alla funzione
                create_and_add_embed_button_to_list(
                    main_window, 
                    embed_data['name'], 
                    embedding_store  # Passa l'intero embedding_store
                )

    main_window.loaded_embedding_filename = embedding_filename or main_window.loaded_embedding_filename

def save_embeddings_to_file(main_window: 'MainWindow', save_as=False):
    if not main_window.merged_embeddings:
        common_widget_actions.create_and_show_messagebox(main_window, 'Embeddings List Empty!', 'No Embeddings available to save', parent_widget=main_window)
        return

    # Definisce il nome del file di salvataggio
    embedding_filename = main_window.loaded_embedding_filename
    if not embedding_filename or save_as:
        embedding_filename, _ = QtWidgets.QFileDialog.getSaveFileName(main_window, filter='JSON (*.json)')

    # Crea una lista di dizionari, ciascuno con il nome dell'embedding e il relativo embedding_store
    embeddings_list = [
        {
            'name': embed_button.embedding_name,
            'embedding_store': {k: v.tolist() for k, v in embed_button.embedding_store.items()}  # Converti gli embedding in liste
        }
        for embed_button in main_window.merged_embeddings
    ]

    # Salva su file
    if embedding_filename:
        with open(embedding_filename, 'w') as embed_file:
            embeddings_as_json = json.dumps(embeddings_list, indent=4)  # Salva con indentazione per leggibilità
            embed_file.write(embeddings_as_json)

            # Mostra un messaggio di conferma
            common_widget_actions.create_and_show_toast_message(main_window, 'Embeddings Saved', f'Saved Embeddings to file: {embedding_filename}')

        main_window.loaded_embedding_filename = embedding_filename



def clear_merged_embeddings(main_window: 'MainWindow'):
    main_window.inputEmbeddingsList.clear()
    for embed_button in main_window.merged_embeddings:
        embed_button.deleteLater()
    main_window.merged_embeddings = []

    for target_face in main_window.target_faces:
        target_face.assigned_embed_buttons = {}
        target_face.calculateAssignedInputEmbedding()
    common_widget_actions.refresh_frame(main_window=main_window)