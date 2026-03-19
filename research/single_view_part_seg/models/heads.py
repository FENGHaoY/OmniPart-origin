import torch.nn as nn


class SemanticHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 5) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class EmbeddingHead(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 16) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, emb_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)

