from model.pam import embed, unembed
from torch import cat, Tensor
from torch.functional import F
from torch.nn import Identity, Linear, Module, ModuleList, ReLU, Sequential
from utils import get_mlp_block


class FAFFM(Module):
    def __init__(self, channels: tuple, level: int, gamma: int) -> None:
        super(FAFFM, self).__init__()
        self._level = level
        self._sample_blocks = ModuleList()
        self._weight_blocks = ModuleList()
        for low in range(0, level):
            self._sample_blocks.append(get_mlp_block(in_features=channels[low], out_features=channels[level]))
            self._weight_blocks.append(get_mlp_block(in_features=channels[level], out_features=gamma))
        self._sample_blocks.append(Identity())
        self._weight_blocks.append(get_mlp_block(in_features=channels[level], out_features=gamma))
        for high in range(level + 1, len(channels)):
            self._sample_blocks.append(get_mlp_block(in_features=channels[high], out_features=channels[level]))
            self._weight_blocks.append(get_mlp_block(in_features=channels[level], out_features=gamma))
        self._sa_weight_fc = Linear(in_features=gamma * len(channels), out_features=len(channels), bias=False)
        self._sa_output_fc = Linear(in_features=channels[level], out_features=channels[level], bias=False)
        self._ca_weight_block = Sequential(
            Linear(in_features=channels[level], out_features=channels[level] // gamma, bias=False),
            ReLU(inplace=True),
            Linear(in_features=channels[level] // gamma, out_features=channels[level], bias=False),
        )

    def forward(self, features: list) -> Tensor:  # (B, C, H, W)
        H, W = features[self._level].shape[2:]
        # FSAM
        sampled_features = []
        weights = []
        for low in range(0, self._level):
            up_sampled_feature = F.adaptive_avg_pool2d(input=features[low], output_size=(H, W))
            up_sampled_feature = embed(input=up_sampled_feature)  # (B*L, C)
            up_sampled_feature = self._sample_blocks[low](input=up_sampled_feature)
            sampled_features.append(up_sampled_feature)
            weights.append(self._weight_blocks[low](input=up_sampled_feature))
        feature = embed(input=features[self._level])  # (B*L, C)
        sampled_features.append(feature)
        weights.append(self._weight_blocks[self._level](input=feature))
        for high in range(self._level + 1, len(features)):
            down_sampled_feature = F.interpolate(input=features[high], size=(H, W), mode='nearest')
            down_sampled_feature = embed(input=down_sampled_feature)  # (B*L, C)
            down_sampled_feature = self._sample_blocks[high](input=down_sampled_feature)
            sampled_features.append(down_sampled_feature)
            weights.append(self._weight_blocks[high](input=down_sampled_feature))
        weight = F.softmax(input=self._sa_weight_fc(input=cat(tensors=weights, dim=1)), dim=1)
        output = sum([sampled_features[level] * weight[:, level: level + 1] for level in range(len(features))])
        output = self._sa_output_fc(input=output)
        # FCAM
        output = unembed(input=output, size=(H, W))
        weight = F.adaptive_avg_pool2d(input=output, output_size=1)  # (B, C, 1, 1)
        weight = F.sigmoid(input=self._ca_weight_block(input=weight.squeeze()))
        output = output * weight.reshape(shape=(*weight.shape, 1, 1)).expand_as(other=output)  # (B, C, H, W)
        return output
