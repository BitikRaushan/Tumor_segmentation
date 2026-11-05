import os
import torch
import numpy as np
from torch.utils.data import Dataset
from preprocessing.pfwf_filter import ProbabilisticWienerFilter

class LiverMRIDataset(Dataset):
    def __init__(self, processed_dir: str, file_list: list, apply_pfwf: bool = True):
        self.processed_dir = processed_dir
        self.file_list = file_list
        self.apply_pfwf = apply_pfwf
        self.filter = ProbabilisticWienerFilter() if apply_pfwf else None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.processed_dir, self.file_list[idx])
        data = np.load(file_path)
        img = data["image"]  # [224, 224] in [0, 1]
        mask = data["mask"]   # [224, 224] in {0, 1}

        if self.apply_pfwf:
            img = self.filter(img)

        img_tensor = torch.from_numpy(img).unsqueeze(0).float()   # [1, 224, 224]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() # [1, 224, 224]

        return img_tensor, mask_tensor