import numpy as np
import torch
import cv2

class PhotoMetricDistortionMultiViewImage:
    def __call__(self, data):
        # Add random brightness, contrast etc.
        images = []
        for img in data['img']:
            img = cv2.convertScaleAbs(img, alpha=np.random.uniform(0.8,1.2), beta=np.random.randint(-10,10))
            images.append(img)
        data['img'] = images
        return data

class NormalizeMultiviewImage:
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)
    def __call__(self, data):
        images = []
        for img in data['img']:
            img = (img - self.mean) / self.std
            images.append(img)
        data['img'] = images
        return data

class PadMultiViewImage:
    def __init__(self, size_divisor):
        self.size_divisor = size_divisor
    def __call__(self, data):
        images = []
        for img in data['img']:
            h, w = img.shape[:2]
            new_h = (h + self.size_divisor - 1) // self.size_divisor * self.size_divisor
            new_w = (w + self.size_divisor - 1) // self.size_divisor * self.size_divisor
            pad_img = np.zeros((new_h, new_w, 3), dtype=img.dtype)
            pad_img[:h, :w] = img
            images.append(pad_img)
        data['img'] = images
        return data

class CustomCollect3D:
    def __init__(self, keys):
        self.keys = keys
    def __call__(self, data):
        return {k: data[k] for k in self.keys if k in data}

def PETRFormatBundle3D(class_names, collect_keys):
    # Compose function
    def transform(data):
        # Format and type conversions
        data['gt_labels_3d'] = torch.tensor(data['gt_labels_3d'], dtype=torch.long)
        data['input_ids'] = torch.tensor(data['input_ids'], dtype=torch.long)
        data['ego_fut_trajs'] = torch.tensor(data['ego_fut_trajs'], dtype=torch.float32)
        return data
    return transform
