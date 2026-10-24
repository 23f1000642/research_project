"""Batched, on-device augmentation for uint8 image tensors.

Images are stored as uint8 [N, C, H, W]. Augmentation (random crop with zero padding,
horizontal flip) is applied per batch on the GPU, so buffer samples can be stored
un-augmented and augmented again every time they are replayed (as in DER/Mammoth).
Normalisation happens inside the model (see models.Normalize).
"""
import torch


class Augmenter:
    def __init__(self, crop_padding: int = 0, hflip: bool = False):
        self.pad = int(crop_padding)
        self.hflip = bool(hflip)

    @staticmethod
    def to_float(x: torch.Tensor) -> torch.Tensor:
        return x.float().div_(255.0) if x.dtype == torch.uint8 else x

    def __call__(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        x = self.to_float(x)
        if not train:
            return x
        if self.pad > 0:
            x = self._random_crop(x)
        if self.hflip:
            flip = torch.rand(x.shape[0], device=x.device) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
        return x

    def _random_crop(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.pad
        padded = torch.nn.functional.pad(x, (p, p, p, p))
        oy = torch.randint(0, 2 * p + 1, (b,), device=x.device)
        ox = torch.randint(0, 2 * p + 1, (b,), device=x.device)
        rows = (oy[:, None] + torch.arange(h, device=x.device))[:, :, None]  # [B,H,1]
        cols = (ox[:, None] + torch.arange(w, device=x.device))[:, None, :]  # [B,1,W]
        bidx = torch.arange(b, device=x.device)[:, None, None]
        out = padded.permute(0, 2, 3, 1)[bidx, rows, cols]  # [B,H,W,C]
        return out.permute(0, 3, 1, 2).contiguous()
