
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision import transforms


class FacialKeypointsDataset(Dataset):

    def __init__(self, csv_path=None, X=None, Y=None, indices=None):
        super().__init__()

        if X is not None and Y is not None:
            # Dataset aus bereits geladenen Daten erstellen
            self.X = X
            self.Y = Y
            self.indices = (
                np.arange(len(X)) if indices is None else indices
            )
        else:
            # CSV einmalig laden
            df = pd.read_csv(csv_path)

            image_strings = df["Image"].values

            keypoint_cols = [
                col for col in df.columns if col != "Image"
            ]

            self.Y = df[keypoint_cols].to_numpy(
                dtype=np.float32
            )

            self.X = np.stack([
                np.fromstring(s, sep=" ", dtype=np.uint8)
                for s in image_strings
            ]).reshape(-1, 96, 96)

            self.indices = np.arange(len(self.X))

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        image = self.transform(self.X[real_idx])
        keypoints = torch.from_numpy(
            self.Y[real_idx].copy()
        )

        return image, keypoints

    def split(self, distribution):
        if (
            len(distribution) != 2
            or any(not isinstance(x, int) or x < 0 for x in distribution)
            or sum(distribution) == 0
        ):
            raise ValueError(
                "distribution muss aus zwei nicht-negativen "
                "Integern bestehen, deren Summe größer als 0 ist."
            )

        total = len(self)
        first_size = int(total * distribution[0] / sum(distribution))

        # Indizes des aktuellen Datasets mischen
        shuffled_indices = self.indices

        first_indices = shuffled_indices[:first_size]
        second_indices = shuffled_indices[first_size:]

        first_dataset = FacialKeypointsDataset(
            X=self.X,
            Y=self.Y,
            indices=first_indices
        )

        second_dataset = FacialKeypointsDataset(
            X=self.X,
            Y=self.Y,
            indices=second_indices
        )

        return first_dataset, second_dataset
