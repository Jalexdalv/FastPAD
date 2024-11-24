from backbone.vgg19 import Vgg19
from torch import load, Tensor
from torch.nn import Module


class FeatureExtractor(Module):
    def __init__(self, backbone: Module, level_map: dict, path: str) -> None:
        super(FeatureExtractor, self).__init__()
        self._backbone = backbone
        self._level_map = level_map
        self._backbone.load_state_dict(state_dict=load(f=path))
        for parameter in self._backbone.parameters():
            parameter.requires_grad = False
        self.channels = []


class Vgg19FeatureExtractor(FeatureExtractor):
    def __init__(self, path: str, levels: tuple) -> None:
        super(Vgg19FeatureExtractor, self).__init__(backbone=Vgg19(),
                                                    level_map={'level_1': 64, 'level_2': 128, 'level_3': 256, 'level_4': 512, 'level_5': 512},
                                                    path=path)
        self._levels = levels
        self._features = self._backbone.features
        self.channels = [self._level_map[level] for level in levels]

    def forward(self, input: Tensor) -> list:
        feature_1_1 = self._features[1](input=self._features[0](input=input))
        feature_1_2 = self._features[3](input=self._features[2](input=feature_1_1))
        feature_2_1 = self._features[6](input=self._features[5](input=self._features[4](input=feature_1_2)))
        feature_2_2 = self._features[8](input=self._features[7](input=feature_2_1))
        feature_3_1 = self._features[11](input=self._features[10](input=self._features[9](input=feature_2_2)))
        feature_3_2 = self._features[13](input=self._features[12](input=feature_3_1))
        feature_3_3 = self._features[15](input=self._features[14](input=feature_3_2))
        feature_3_4 = self._features[17](input=self._features[16](input=feature_3_3))
        feature_4_1 = self._features[20](input=self._features[19](input=self._features[18](input=feature_3_4)))
        feature_4_2 = self._features[22](input=self._features[21](input=feature_4_1))
        feature_4_3 = self._features[24](input=self._features[23](input=feature_4_2))
        feature_4_4 = self._features[26](input=self._features[25](input=feature_4_3))
        feature_5_1 = self._features[29](input=self._features[28](input=self._features[27](input=feature_4_4)))
        feature_5_2 = self._features[31](input=self._features[30](input=feature_5_1))
        feature_5_3 = self._features[33](input=self._features[32](input=feature_5_2))
        feature_5_4 = self._features[35](input=self._features[34](input=feature_5_3))
        feature_map = {'level_1': feature_1_1 + feature_1_2,
                       'level_2': feature_2_1 + feature_2_2,
                       'level_3': feature_3_1 + feature_3_2 + feature_3_3 + feature_3_4,
                       'level_4': feature_4_1 + feature_4_2 + feature_4_3 + feature_4_4,
                       'level_5': feature_5_1 + feature_5_2 + feature_5_3 + feature_5_4}
        return [feature_map[level] for level in self._levels]
