from torch import Tensor
from torch.nn import Linear, Module


class FCM(Module):
    def __init__(self, num_channels: int) -> None:
        super(FCM, self).__init__()
        self._project_fc = Linear(in_features=num_channels, out_features=num_channels, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        return self._project_fc(input=input)
