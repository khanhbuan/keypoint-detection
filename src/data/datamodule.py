from typing import Any, Dict, Optional, Tuple

import os
import cv2
import torch
import hydra
import rootutils
import numpy as np
import albumentations as A
from omegaconf import DictConfig
from lightning import LightningDataModule
from src.data.components.dataset import dataset
from torch.utils.data import DataLoader, Dataset, random_split
from src.data.components.transformed_dataset import transformed_dataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

       #       RED           GREEN          BLACK          CYAN           YELLOW        MAGENTA         GREEN          BLUE 
colors = [[000,000,255], [000,255,000], [000,000,000], [255,255,000], [000,255,255], [255,000,255], [000,255,000], [255,000,000], \
       #      BLACK          CYAN           YELLOW        GREEN           BLUE           CYAN          MAGENTA
          [000,000,000], [255,255,000], [000,255,255], [000,255,000], [255,000,000], [255,255,000], [255,000,255], \
       #      BLUE            GRAY           NAVY           PINK         MAGENTA          CYAN           PINK    
          [255,000,000], [128,128,128], [000,000,128], [203,192,255], [255,  0,255], [255,255,000], [203,192,255]]

def custom_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    return images, targets

class DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        train_val_test_split: Tuple[int, int, int] = (23, 5, 1),
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = False,
        train_transform: Optional[A.Compose] = None,
        val_test_transform: Optional[A.Compose] = None,
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            raw_dataset = dataset(data_dir = self.hparams.data_dir)

            train, val, test = random_split(
                dataset=raw_dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            self.data_train = transformed_dataset(train, transform=self.hparams.train_transform)
            self.data_val = transformed_dataset(val, transform=self.hparams.val_test_transform)
            self.data_test = transformed_dataset(test, transform=self.hparams.val_test_transform)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=custom_collate,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=custom_collate,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=custom_collate,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

@hydra.main(version_base="1.3", config_path="../../configs/data", config_name="data")
def main(cfg: DictConfig) -> Optional[float]:
    datamodule: LightningDataModule = hydra.utils.instantiate(config=cfg)
    datamodule.setup()
    for idx, batch in enumerate(datamodule.train_dataloader()):
        img, target = batch
        box, label, keypoint = target[0]["boxes"], target[0]["labels"], target[0]["keypoints"]

        box, label, keypoint = box[0], label[0], keypoint[0]

        img = np.array(torch.permute(img[0], (1, 2, 0)), dtype=np.float32)

        for i in range(len(keypoint)):
            x = int(keypoint[i][0])
            y = int(keypoint[i][1])
            cv2.circle(img, (x, y), radius=2, thickness=-1, color=colors[i])

        x1, y1, x2, y2 = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color = colors[0])

        cv2.imwrite(os.path.join('visualization', 'output' + str(idx) + '.png'), img)

if __name__ == "__main__":
    main()