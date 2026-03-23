"""Backward-compatible fusion trainer wrapper."""

from training.train_fusion import train_fusion


class FusionTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self):
        return train_fusion(self.cfg)
