import torch

def custom_collate(batch):
    batched = {}
    for key in batch[0]:
        try:
            batched[key] = torch.stack([b[key] for b in batch])
        except Exception:
            batched[key] = [b[key] for b in batch]
    return batched
