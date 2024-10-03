# Script Usage Example
# 'python3 convert_old_rope_embeddings.py old_merged_embeddings.txt new_merged_embeddings.json'  

import sys
import json
import numpy as np

input_filename = sys.argv[1]
try:
    output_filename = sys.argv[2]
except:
    output_filename = f'{input_filename.split(".")[0]}_converted.json'
temp0 = []
new_embed_list = []
with open(input_filename, "r") as embedfile:
    old_data = embedfile.read().splitlines()

    for i in range(0, len(old_data), 513):
        new_embed_data = {'name': old_data[i][6:], 'embedding': old_data[i+1:i+513]}
        for i, val in enumerate(new_embed_data['embedding']):
            new_embed_data['embedding'][i] = float(val)
        new_embed_list.append(new_embed_data)

with open(output_filename, 'w') as embed_file:
    embeds_data = json.dumps(new_embed_list)
    embed_file.write(embeds_data)