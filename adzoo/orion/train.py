import torch
from torch.utils.data import DataLoader
from adzoo.orion.datasets.b2d_orion_dataset import B2DOrionDataset
from adzoo.orion.models.orion_model import OrionModel
from adzoo.orion.losses.l1_loss import L1Loss
from adzoo.orion.losses.probabilistic_loss import ProbabilisticLoss
from adzoo.orion.pipelines.collate import custom_collate
from adzoo.orion.configs.orion_stage1_train import data, model, optimizer as optimizer_cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    train_cfg = data['train']
    train_set = B2DOrionDataset(
        ann_file=train_cfg["ann_file"],
        data_root=train_cfg["data_root"],
        map_file=train_cfg["map_file"],
        pipeline=train_cfg["pipeline"],
        classes=train_cfg["classes"],
        modality=train_cfg["modality"],
        queue_length=train_cfg["queue_length"],
        past_frames=train_cfg["past_frames"],
        future_frames=train_cfg["future_frames"],
    )
    train_loader = DataLoader(
        train_set, batch_size=data["samples_per_gpu"], 
        shuffle=True, collate_fn=custom_collate, num_workers=data["workers_per_gpu"]
    )

    model_instance = OrionModel().to(device)
    optimizer = torch.optim.AdamW(model_instance.parameters(), lr=optimizer_cfg["lr"], weight_decay=optimizer_cfg["weight_decay"])
    loss_l1 = L1Loss(loss_weight=model["loss_plan_reg"]["loss_weight"])
    loss_prob = ProbabilisticLoss(loss_weight=model["loss_vae_gen"]["loss_weight"])

    for epoch in range(6):  # num_epochs as in config
        model_instance.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            output = model_instance(batch["img"], batch["input_ids"])
            l1 = loss_l1(output, batch["ego_fut_trajs"])
            nll = loss_prob(output, torch.zeros_like(output), batch["ego_fut_trajs"])
            loss = l1 + nll
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                print(f"Epoch {epoch} Iter {i} Loss: {loss.item()}")

if __name__ == "__main__":
    main()
