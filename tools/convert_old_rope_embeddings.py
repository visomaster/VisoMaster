# Script Usage Example
# 'python3 convert_old_rope_embeddings.py old_merged_embeddings.txt new_merged_embeddings.json'  

import sys
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser("Rope Embeddings Converter")
parser.add_argument("old_embeddings_file", help="Old Embeddings File", type=str)
parser.add_argument("--output_embeddings_file", help="New Embeddings File", type=str)
parser.add_argument("--recognizer_model", help="Face Recognizer Model using which the embedding was created", default='Inswapper128ArcFace', choices=('Inswapper128ArcFace', 'SimSwapArcFace', 'GhostArcFace', 'GhostArcFace', 'GhostArcFace'), type=str)
args = parser.parse_args()
input_filename = args.old_embeddings_file

output_filename = args.output_embeddings_file or f'{input_filename.split(".")[0]}_converted.json'

recognizer_model = args.recognizer_model
temp0 = []
new_embed_list = []
with open(input_filename, "r") as embedfile:
    old_data = embedfile.read().splitlines()

    for i in range(0, len(old_data), 513):
        new_embed_data = {'name': old_data[i][6:], 'embedding_store': {recognizer_model: old_data[i+1:i+513]}}
        for i, val in enumerate(new_embed_data['embedding_store'][recognizer_model]):
            new_embed_data['embedding_store'][recognizer_model][i] = float(val)
        new_embed_list.append(new_embed_data)

with open(output_filename, 'w') as embed_file:
    embeds_data = json.dumps(new_embed_list)
    embed_file.write(embeds_data)