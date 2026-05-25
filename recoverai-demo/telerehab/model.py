import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class ResidualTCNBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = self.downsample(x)
        return self.out_relu(out + res)


class ConditionedTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_exercises: int = 6,
        ex_embed_dim: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_exercises = int(num_exercises)
        self.ex_embed_dim = int(ex_embed_dim)

        self.exercise_embedding = nn.Embedding(num_exercises, ex_embed_dim)
        self.backbone = nn.Sequential(
            ResidualTCNBlock(input_dim + ex_embed_dim, 64, kernel_size=3, dilation=1, dropout=dropout),
            ResidualTCNBlock(64, 64, kernel_size=3, dilation=2, dropout=dropout),
            ResidualTCNBlock(64, 96, kernel_size=3, dilation=4, dropout=dropout),
            ResidualTCNBlock(96, 96, kernel_size=3, dilation=8, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, exercise_idx: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        ex_emb = self.exercise_embedding(exercise_idx).unsqueeze(-1).expand(-1, -1, T)
        x = torch.cat([x, ex_emb], dim=1)
        feats = self.backbone(x)
        pooled = self.pool(feats)
        logits = self.head(pooled).squeeze(-1)
        return logits
class ExerciseTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_exercises: int = 6,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_exercises = int(num_exercises)

        self.backbone = nn.Sequential(
            ResidualTCNBlock(input_dim, 64, kernel_size=3, dilation=1, dropout=dropout),
            ResidualTCNBlock(64, 64, kernel_size=3, dilation=2, dropout=dropout),
            ResidualTCNBlock(64, 96, kernel_size=3, dilation=4, dropout=dropout),
            ResidualTCNBlock(96, 96, kernel_size=3, dilation=8, dropout=dropout),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_exercises),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        pooled = self.pool(feats)
        logits = self.head(pooled)
        return logits
