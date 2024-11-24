from torch import Tensor
from torch.functional import F
from torch.nn import Linear, Module, Sequential
from utils import get_mlp_block


class PDM(Module):
    def __init__(self, in_channels: int, betas: tuple) -> None:
        super(PDM, self).__init__()
        self._classifier = Sequential()
        for beta in betas:
            out_channels = in_channels // beta
            self._classifier.append(get_mlp_block(in_features=in_channels, out_features=out_channels))
            in_channels = out_channels
        self._classifier.append(Linear(in_features=in_channels, out_features=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input=self._classifier(input=input))
