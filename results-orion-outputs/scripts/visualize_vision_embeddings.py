import numpy as np

folder = 'results-orion-outputs'

# Object Embeddings
obj_embed = np.load(f'{folder}/vision_obj_embeddings.npy')
print("Object Embedding Shape:", obj_embed.shape)
print("Object Embedding Array:\n", obj_embed)

# Map Embeddings
map_embed = np.load(f'{folder}/vision_map_embeddings.npy')
print("\nMap Embedding Shape:", map_embed.shape)
print("Map Embedding Array:\n", map_embed)

