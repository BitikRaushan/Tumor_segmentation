import os
import json
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.full_model import BR_UA_MDASGTNet
from datasets.liver_dataset import LiverMRIDataset
from losses import CombinedLoss
from evaluation.metrics import compute_batch_metrics

def main():
    with open("configs/base_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)

    with open(cfg["data"]["splits_file"], "r") as f:
        splits = json.load(f)

    fold_key = f"fold_{cfg['data']['selected_fold']}"
    train_files = splits[fold_key]["train"]
    val_files = splits[fold_key]["val"]

    train_ds = LiverMRIDataset(cfg["data"]["processed_dir"], train_files, apply_pfwf=True)
    val_ds = LiverMRIDataset(cfg["data"]["processed_dir"], val_files, apply_pfwf=True)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)

    model = BR_UA_MDASGTNet(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        mc_dropout_rate=cfg["model"]["mc_dropout_rate"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    criterion = CombinedLoss(
        l_focal=cfg["loss_weights"]["lambda_focal"],
        l_dice=cfg["loss_weights"]["lambda_dice"],
        l_bdry=cfg["loss_weights"]["lambda_boundary"]
    )

    best_val_dice = 0.0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['training']['epochs']}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation Step
        model.eval()
        val_dices = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                res = compute_batch_metrics(logits, masks)
                val_dices.append(res["dice"])

        mean_dice = sum(val_dices) / max(len(val_dices), 1)
        print(f"Epoch {epoch} Result -> Train Loss: {train_loss/len(train_loader):.4f} | Val Dice: {mean_dice:.4f}")

        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            save_path = os.path.join(cfg["training"]["checkpoint_dir"], "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} (Dice: {best_val_dice:.4f})")

if __name__ == "__main__":
    main()