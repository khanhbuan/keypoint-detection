import os
import cv2
import json
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        anno_path = os.path.join(data_dir, 'addition/default.json')
        with open(anno_path, 'r') as file:
            data = json.load(file)
        
        self.anno = data["items"]
    
    def __len__(self):
        return len(self.anno)
    
    def __getitem__(self, idx):
        sample = self.anno[idx]
        img_path = os.path.join(self.data_dir, "images/default", sample["id"] + ".jpg")

        keypoints = []
        points_list=[]
        points_list2=[]

        for sampl in sample["annotations"]:
            if sampl["type"] == "polyline":
                if len(sampl["points"]) == 6:
                    points_list2 = sampl["points"]
                elif len(sampl["points"]) == 44:
                    points_list = sampl["points"]

        keypoints.append([points_list2[0], points_list2[1]])
        
        for id in range(int(len(points_list) / 2)):
            keypoints.append([points_list[2*id], points_list[2*id+1]])
        
        keypoints.append([points_list2[2], points_list2[3]])
        keypoints.append([points_list2[4], points_list2[5]])

        return img_path, keypoints
    
if __name__ == "__main__":
    dataset = dataset("./data")