import torch
import numpy as np
from medpy import metric

def compute_batch_metrics(pred_logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(pred_logits).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    preds = (probs > threshold).astype(np.uint8)
    targets = (targets > 0.5).astype(np.uint8)

    dice_list, iou_list, hd95_list, asds_list = [], [], [], []

    for i in range(preds.shape[0]):
        p = preds[i, 0]
        t = targets[i, 0]

        if np.sum(t) == 0 and np.sum(p) == 0:
            dice_list.append(1.0)
            iou_list.append(1.0)
            continue
        elif np.sum(t) == 0 or np.sum(p) == 0:
            dice_list.append(0.0)
            iou_list.append(0.0)
            continue

        d = metric.binary.dc(p, t)
        j = metric.binary.jc(p, t)
        dice_list.append(d)
        iou_list.append(j)

        try:
            hd = metric.binary.hd95(p, t)
            asd = metric.binary.asd(p, t)
            hd95_list.append(hd)
            asds_list.append(asd)
        except Exception:
            pass

    return {
        "dice": float(np.mean(dice_list)) if dice_list else 0.0,
        "iou": float(np.mean(iou_list)) if iou_list else 0.0,
        "hd95": float(np.mean(hd95_list)) if hd95_list else 0.0,
        "assd": float(np.mean(asds_list)) if asds_list else 0.0
    }