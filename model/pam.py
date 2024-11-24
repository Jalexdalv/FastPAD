from math import floor
from torch import Tensor
from torch.functional import F
from torch.nn import Module


class PAM(Module):
    def __init__(self, patch_size: tuple, patch_stride: tuple) -> None:
        super(PAM, self).__init__()
        self._patch_size = patch_size
        self._patch_stride = patch_stride

    def aggregate(self, input: Tensor) -> Tensor:
        B, C, H, W = input.shape
        patched_size = floor((H - self._patch_size[0]) / self._patch_stride[0]) + 1, floor((W - self._patch_size[1]) / self._patch_stride[1]) + 1
        G = self._patch_size[0] * self._patch_size[1]
        L = patched_size[0] * patched_size[1]
        output = F.unfold(input=input, kernel_size=self._patch_size, stride=self._patch_stride)  # (B, C*G, L)
        output = output.reshape(shape=(B, C, G, L))  # (B, C, G, L)
        output = output.permute(dims=(0, 1, 3, 2))  # (B, C, L, G)
        output = output.mean(dim=3)  # (B, C, L)
        output = output.reshape(shape=(B, C, *patched_size))  # (B, C, patched_H, patched_W)
        return output


def embed(input: Tensor) -> Tensor:
    B, C, H, W = input.shape
    L = H * W
    P = B * L
    output = input.reshape(shape=(B, C, L))  # (B, C, L)
    output = output.permute(dims=(0, 2, 1))  # (B, L, C)
    output = output.reshape(shape=(P, C))  # (P, C)
    return output


def unembed(input: Tensor, size: tuple) -> Tensor:
    P, C = input.shape
    L = size[0] * size[1]
    B = P // L
    output = input.reshape(shape=(B, L, C))  # (B, L, C)
    output = output.permute(dims=(0, 2, 1))  # (B, C, L)
    output = output.reshape(shape=(B, C, *size))  # (B, C, H, W)
    return output
