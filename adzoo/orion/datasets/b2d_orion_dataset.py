import os
import pickle
import torch
import numpy as np
import cv2

class B2DOrionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_file,
        data_root,
        map_file,
        pipeline,
        classes,
        modality,
        queue_length=1,
        past_frames=2,
        future_frames=6,
    ):
        with open(ann_file, "rb") as f:
            self.ann_data = pickle.load(f)
        self.data_root = data_root
        self.map_file = map_file
        self.pipeline = pipeline
        self.classes = classes
        self.modality = modality
        self.queue_length = queue_length
        self.past_frames = past_frames
        self.future_frames = future_frames
        # Preload map data if required (adapt for your format)
        self.map_data = np.load(self.map_file, allow_pickle=True) if map_file else None

    def load_multiview_images(self, sample):
        images = []
        if "cams" in sample:
            for cam in sample["cams"]:
                # Compose absolute image path as done in Orion repo
                img_path = os.path.join(self.data_root, cam["img_path"])
                img = cv2.imread(img_path)
                # If image is grayscale, expand to rgb
                if img is not None and len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                images.append(img)
        else:
            # Fallback: assume single image path
            img_path = os.path.join(self.data_root, sample["img_path"])
            img = cv2.imread(img_path)
            if img is not None and len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            images.append(img)
        # Convert images to np array as expected for pipelines
        images = np.stack(images, axis=0)
        return images

    def __len__(self):
        return len(self.ann_data)

    def __getitem__(self, idx):
        sample = self.ann_data[idx]
        images = self.load_multiview_images(sample)
        
        # Load map, labels, trajectories and additional fields
        traj = np.asarray(sample.get("ego_fut_trajs", np.zeros((self.future_frames, 2))))
        commands = np.asarray(sample.get("ego_fut_cmd", np.zeros((self.future_frames))))
        labels = np.asarray(sample.get("labels", []), dtype=np.int64)
        input_ids = np.asarray(sample.get("input_ids", []), dtype=np.int64)
        
        data = {
            "img": images,
            "ego_fut_trajs": traj,
            "ego_fut_cmd": commands,
            "gt_labels_3d": labels,
            "input_ids": input_ids,
        }

        # Run through all transforms in pipeline (just like Orion repo)
        for tform in self.pipeline:
            data = tform(data)
        return data
