import numpy as np

folder = 'results-orion-outputs'
traj = np.load(f'{folder}/ego_fut_preds_vae.npy')
print("\nTrajectory Shape:", traj.shape)
print("Trajectory Array:\n", traj)

