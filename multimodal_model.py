import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class MultimodalModel(nn.Module):
    def __init__(self, tabular_dim):
        super().__init__()

        # 🖼️ Pretrained CNN (feature extractor)
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()

        # 🔒 Freeze CNN weights (important for CPU)
        for param in self.cnn.parameters():
            param.requires_grad = False

        # 🔢 Tabular feature network
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU()
        )

        # 🔗 Fusion + Regression head
        self.regressor = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, tabular):
        img_feat = self.cnn(image)
        tab_feat = self.tabular_net(tabular)
        x = torch.cat([img_feat, tab_feat], dim=1)
        return self.regressor(x).squeeze()

