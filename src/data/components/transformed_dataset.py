import cv2
import torch
import numpy as np
import albumentations as A
from src.data.components.dataset import dataset
from torch.utils.data import Dataset

       #       RED           GREEN          BLACK          CYAN           YELLOW        MAGENTA         GREEN          BLUE 
colors = [[000,000,255], [000,255,000], [000,000,000], [255,255,000], [000,255,255], [255,000,255], [000,255,000], [255,000,000], \
       #      BLACK          CYAN           YELLOW        GREEN           BLUE           CYAN          MAGENTA          CYAN
          [000,000,000], [255,255,000], [000,255,255], [000,255,000], [255,000,000], [255,255,000], [255,000,255], [255,255,000], \
       #      BLUE            GRAY           NAVY           PINK         MAGENTA          CYAN           PINK          YELLOW          GREEN
          [255,000,000], [128,128,128], [000,000,128], [203,192,255], [255,000,255], [255,255,000], [203,192,255], [000,255,255], [000,255,000]]

class transformed_dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, idx):
        img_path, keypoints = self.dataset.__getitem__(idx)

        img = cv2.imread(img_path)
        h, w, _ = img.shape
        center = w/2
        img = img[0:h, int(center - h/2):int(center + h/2)]

        for i in range(len(keypoints)):
            keypoints[i][0] -= int(center - h/2)
 
        h, w, _ = img.shape
        x_max, y_max, x_min, y_min = 0, 0, w, h
        for keypoint in keypoints:
            if keypoint[0] < 0 or keypoint[1] < 0:
                continue

            x_max = max(x_max, keypoint[0])
            y_max = max(y_max, keypoint[1])

            x_min = min(x_min, keypoint[0])
            y_min = min(y_min, keypoint[1])
        offset = 20
        boxes = [[w/2-abs(w/2 - x_min)-offset, h/2-abs(h/2 - y_min)-offset, w/2+abs(w/2 - x_max)+offset, h/2+abs(h/2-y_max)+offset]]
        labels = ["keypoint"]

        if self.transform is not None:
            transformed = self.transform(image=img, keypoints=keypoints, bboxes=boxes, labels = labels)
            img, keypoints, boxes = transformed["image"], transformed["keypoints"], transformed["bboxes"]
        
        h, w, _ = img.shape

        # keypoints = keypoints / np.array([w, h])
        # boxes = boxes / np.array([w, h, w, h])

        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        visible = torch.zeros(keypoints.shape[0], 1)        
        for id, point in enumerate(keypoints):
            if point[0] != 0 and point[1] != 0:
                visible[id] = 1

        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)

        keypoints = torch.concat([keypoints, visible], dim = 1)[None,:,:]
        boxes = torch.tensor(boxes[0])[None,:]
        labels = torch.tensor([1], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "keypoints": keypoints}
        
        return img, target

if __name__ == "__main__":
    train_transform = A.Compose([
        A.Resize(height=700, width=700),
        A.CenterCrop(height=384, width=384),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0, rotate_limit=0, p=1.0),
        A.RandomScale(scale_limit=0.1, p=1.0),
        A.Rotate(limit=90, p=1.0),
        A.CoarseDropout(max_holes=18, max_height=18, max_width=18)
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels']), keypoint_params = A.KeypointParams(format= "xy", remove_invisible=False))

    val_transform = A.Compose([
        A.Resize(height=700, width=700),
        # A.CenterCrop(height=384, width=384),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels']), keypoint_params = A.KeypointParams(format= "xy", remove_invisible=False))
    
    dataset = transformed_dataset(dataset=dataset("./data"), transform=val_transform)

    for idx in range(dataset.__len__()):
        img, target = dataset.__getitem__(idx)

        box, label, keypoint = target["boxes"], target["labels"], target["keypoints"]

        box, label, keypoint = box[0], label[0], keypoint[0]

        img = np.array(torch.permute(img, (1, 2, 0)), dtype=np.float32)

        for i in range(len(keypoint)):
            x = int(keypoint[i][0])
            y = int(keypoint[i][1])
            cv2.circle(img, (x, y), radius=2, thickness=-1, color=colors[i])

        # x1, y1, x2, y2 = box
        # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color = colors[0])

        cv2.imwrite('visualization/output' + str(idx) + ".png", img)
        
