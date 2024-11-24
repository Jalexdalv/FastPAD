from model.pam import embed, unembed
from torch import cat, Tensor
from torch.functional import F
from torch.nn import Linear, Module, ReLU, Sequential


class ADWF(Module):
    def __init__(self, num_levels: int) -> None:
        super(ADWF, self).__init__()
        self._wgb = Sequential(
            Linear(in_features=num_levels, out_features=num_levels, bias=False),
            ReLU(inplace=True),
            Linear(in_features=num_levels, out_features=num_levels, bias=False)
        )

    def forward(self, scores: list) -> Tensor:
        scores = [F.interpolate(input=score, size=scores[0].shape[2:], mode='nearest') for score in scores]
        output = embed(input=cat(tensors=scores, dim=1))
        weight = F.softmax(input=self._wgb(input=output), dim=1)
        # output = weight.sum(dim=1, keepdim=True) / (weight / output).sum(dim=1, keepdim=True)
        output = ((output ** (1 - weight)).prod(dim=1, keepdim=True)) ** (1 / (1 - weight).sum(dim=1, keepdim=True))
        # output = (output * weight).prod(dim=1, keepdim=True)
        output = unembed(input=output, size=scores[0].shape[2:])
        return output
