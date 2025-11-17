import numpy as np

folder = 'results-orion-outputs'
bbox_results = np.load(f'{folder}/bbox_results.npy', allow_pickle=True)
print("\nBounding Box Results Type:", type(bbox_results))
if hasattr(bbox_results, 'shape'):
    print("Bounding Box Results Shape:", bbox_results.shape)
print("Bounding Box Results Array:\n", bbox_results)

lane_results = np.load(f'{folder}/lane_results.npy', allow_pickle=True)
print("\nLane Results Type:", type(lane_results))
if hasattr(lane_results, 'shape'):
    print("Lane Results Shape:", lane_results.shape)
print("Lane Results Array:\n", lane_results)

