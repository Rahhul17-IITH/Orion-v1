import numpy as np

folder = 'results-orion-outputs'
ego_feature = np.load(f'{folder}/ego_feature.npy')
print("\nEgo Feature Shape:", ego_feature.shape)
print("Ego Feature Array:\n", ego_feature)

