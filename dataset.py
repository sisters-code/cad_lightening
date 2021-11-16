from pathlib import Path
import random
from typing import Any, Dict, Optional, Sequence, Union

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import gamma_transform, calculate_average_brightness


class FaceDataset4Crops(Dataset):
    def __init__(
        self,
        dataset_root: Union[str, Path],
        split_file: Union[str, Path],
        transform = None,
        balance_samples: bool = False,
        image_names: Sequence[str] = ('1.JPG',)
    ) -> None:
        super().__init__()

        self.transform = transform
        self.image_names = image_names

        with open(split_file, 'r') as f:
            lines = f.read().strip().splitlines()
        lines = [ln.split(' ') for ln in lines]
        self.samples = [
            (Path(dataset_root) / 'images' / n, int(i)) for n, i in lines]

        if balance_samples:
            neg_samples = [s for s in self.samples if s[1] == 0]
            pos_samples = [s for s in self.samples if s[1] == 1]
            if len(neg_samples) < len(pos_samples):
                mult = len(pos_samples) // len(neg_samples) + 1
                neg_samples = neg_samples * mult
                neg_samples = neg_samples[:len(pos_samples)]
            else:
                mult = len(neg_samples) // len(pos_samples) + 1
                pos_samples = pos_samples * mult
                pos_samples = pos_samples[:len(neg_samples)]
            self.samples = pos_samples + neg_samples

        neg_samples = [s for s in self.samples if s[1] == 0]
        pos_samples = [s for s in self.samples if s[1] == 1]
        print(f'Loaded {len(self.samples)} samples, being {len(pos_samples)} positives and {len(neg_samples)} negatives.')

    def __getitem__(self, index) -> Dict[str, Any]:
        img = [cv.imread(str(self.samples[index][0] / name)) for name in self.image_names]
        img = np.concatenate(img, axis=2)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255
        if self.transform is not None:
            img = self.transform(img)

        label = self.samples[index][1]

        return {
            'image': img,
            'label': label,
            'path': str(self.samples[index][0])
        }

    def __len__(self):
        return len(self.samples)


class MultiFacePartDataset(Dataset):
    def __init__(
        self,
        dataset_root: Union[str, Path],
        split_file: Union[str, Path],
        transform = None,
        balance_samples: bool = False,
        part_names: Sequence[str] = ('head',),
        use_gray: bool = False,
        use_eqhist: bool = False,
        use_gamma: bool = False
    ) -> None:
        super().__init__()

        self.transform = transform
        self.part_names = part_names
        self.use_gray = use_gray
        self.use_eqhist = use_eqhist
        self.use_gamma = use_gamma

        with open(split_file, 'r') as f:
            lines = f.read().strip().splitlines()
        lines = [ln.split(' ') for ln in lines]
        names = [ln[0].split('/')[1] for ln in lines]
        labels = [int(ln[1]) for ln in lines]
        img_paths = {name: Path(dataset_root).glob(f'**/{name}.JPG') for name in part_names}
        
        self.samples = []
        n = 0
        used_names = [0] * len(names)
        for imp in img_paths[part_names[0]]:
            try:
                i = names.index(imp.parent.name)
                n += 1
                used_names[i] = 1
                samp = {'label': labels[i], part_names[0]: imp}
                valid = True
                for name in part_names[1:]:
                    samp[name] = imp.parent / f'{name}.JPG'
                    if not samp[name].exists():
                        valid = False
                        break

                if valid:
                    self.samples.append(samp)
            except ValueError:
                continue

        if balance_samples:
            neg_samples = [s for s in self.samples if s['label'] == 0]
            pos_samples = [s for s in self.samples if s['label'] == 1]
            if len(neg_samples) < len(pos_samples):
                mult = len(pos_samples) // len(neg_samples) + 1
                neg_samples = neg_samples * mult
                neg_samples = neg_samples[:len(pos_samples)]
            else:
                mult = len(neg_samples) // len(pos_samples) + 1
                pos_samples = pos_samples * mult
                pos_samples = pos_samples[:len(neg_samples)]
            self.samples = pos_samples + neg_samples
            # random.shuffle(self.samples)

        neg_samples = [s for s in self.samples if s['label'] == 0]
        pos_samples = [s for s in self.samples if s['label'] == 1]
        print(f'Loaded {len(self.samples)} samples, being {len(pos_samples)} positives and {len(neg_samples)} negatives.')

    def __getitem__(self, index) -> Dict[str, Any]:
        out = {}
        for k, v in self.samples[index].items():
            if k == 'label':
                out['label'] = v
            else:
                img = cv.imread(str(v))
                if self.use_gamma:
                    lightness, _, _, _ = calculate_average_brightness(img)
                    if lightness >= 190:
                        img = gamma_transform(img, 6)
                if self.use_gray:
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    if self.use_eqhist:
                        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        img = clahe.apply(img)
                        img = cv.equalizeHist(img)
                    img = img[..., None]
                img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255
                out[k] = img
                out['path'] = str(v)

        if self.transform is not None:
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    out[k] = self.transform(v)

        return out

    def __len__(self):
        return len(self.samples)
