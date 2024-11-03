import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torch import nn

class backbone(nn.Module):
    def __init__(self):
        super().__init__()
        num_keypoints=22
        
        self.model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers=1)
        in_features = self.model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
        self.model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features, num_keypoints)
        
    def forward(self, img, target):
        return self.model(img, target)

if __name__ == "__main__":
    model = backbone()