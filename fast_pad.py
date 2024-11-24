from model.pam import embed, unembed
from kornia.filters import gaussian_blur2d
from numpy import ndarray
from torch import cat, Tensor, zeros_like
from torch.functional import F
from torch.nn import Module
from utils import convert_tensor_to_numpy


class FastPAD(Module):
    def __init__(self, settings: dict) -> None:
        super(FastPAD, self).__init__()
        self._feature_extractor = settings['feature_extractor']
        self._pam = settings['pam'] if 'pam' in settings else None
        self._faffms = settings['faffms'] if 'faffms' in settings else None
        self._fcms = settings['fcms'] if 'fcms' in settings else None
        self._pdms = settings['pdms']
        self._adwf = settings['adwf']
        self._image_size = settings['image_size']
        self._sigma = settings['sigma']

    def compute_loss(self, input: Tensor, augmented_input: Tensor, mask: Tensor) -> Tensor:
        output, augmented_output = self(input=input, augmented_input=augmented_input)
        return F.binary_cross_entropy(input=output, target=zeros_like(input=output)) + F.binary_cross_entropy(input=augmented_output, target=mask)

    def compute_score(self, input: Tensor) -> ndarray:
        output = self(input=input)
        output = gaussian_blur2d(input=output, kernel_size=(self._sigma * 6 + 1, self._sigma * 6 + 1), sigma=(self._sigma, self._sigma))
        output = convert_tensor_to_numpy(tensor=output.squeeze())
        return output

    def forward(self, input: Tensor, augmented_input: Tensor = None):
        output = self._forward(input=input)
        if augmented_input is not None:
            augmented_output = self._forward(input=augmented_input)
            return output, augmented_output
        return output

    def _forward(self, input: Tensor) -> Tensor:
        features = self._feature_extractor(input=input)
        if self._pam is not None:
            features = [self._pam.aggregate(input=feature) for feature in features]

        # for i in range(0, 4):
        #     feature = features[i]
        #     import cv2, numpy, utils
        #     image0 = feature[0].clone().mean(dim=0)
        #     image0 = (image0 - image0.min()) / (image0.max() - image0.min())
        #     image0 = image0.unsqueeze(0).repeat_interleave(repeats=3, dim=0)
        #     image0 = numpy.uint8(numpy.around(a=utils.convert_tensor_to_numpy(tensor=image0.permute(dims=(1, 2, 0))) * 255))
        #     image0 = cv2.applyColorMap(src=image0, colormap=cv2.COLORMAP_JET)
        #     image0 = cv2.resize(src=image0, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        #     cv2.imwrite(filename='pam/{}.png'.format(i), img=image0)

        if self._faffms is not None:
            features = [faffm(features=features) for faffm in self._faffms]

        # for i in range(0, 4):
        #     feature = features[i]
        #     import cv2, numpy, utils
        #     image0 = feature[0].clone().mean(dim=0)
        #     image0 = (image0 - image0.min()) / (image0.max() - image0.min())
        #     image0 = image0.unsqueeze(0).repeat_interleave(repeats=3, dim=0)
        #     image0 = numpy.uint8(numpy.around(a=utils.convert_tensor_to_numpy(tensor=image0.permute(dims=(1, 2, 0))) * 255))
        #     image0 = cv2.applyColorMap(src=image0, colormap=cv2.COLORMAP_JET)
        #     image0 = cv2.resize(src=image0, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        #     cv2.imwrite(filename='faffm/{}.png'.format(i), img=image0)

        embeddings = [embed(input=feature) for feature in features]
        if self._fcms is not None:
            embeddings = [fcm(input=embedding) for fcm, embedding in zip(self._fcms, embeddings)]

        # for i in range(0, 4):
        #     feature = unembed(input=embeddings[i], size=features[i].shape[2:])
        #     import cv2, numpy, utils
        #     image0 = feature[0].clone().mean(dim=0)
        #     image0 = (image0 - image0.min()) / (image0.max() - image0.min())
        #     image0 = image0.unsqueeze(0).repeat_interleave(repeats=3, dim=0)
        #     image0 = numpy.uint8(numpy.around(a=utils.convert_tensor_to_numpy(tensor=image0.permute(dims=(1, 2, 0))) * 255))
        #     image0 = cv2.applyColorMap(src=image0, colormap=cv2.COLORMAP_JET)
        #     image0 = cv2.resize(src=image0, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        #     cv2.imwrite(filename='fcm/{}.png'.format(i), img=image0)

        scores = [pdm(input=embedding) for pdm, embedding in zip(self._pdms, embeddings)]
        scores = [unembed(input=score, size=feature.shape[2:]) for score, feature in zip(scores, features)]

        # for i in range(0, 4):
        #     feature = scores[i]
        #     import cv2, numpy, utils
        #     image0 = feature[0].clone().mean(dim=0)
        #     image0 = (image0 - image0.min()) / (image0.max() - image0.min())
        #     image0 = image0.unsqueeze(0).repeat_interleave(repeats=3, dim=0)
        #     image0 = numpy.uint8(numpy.around(a=utils.convert_tensor_to_numpy(tensor=image0.permute(dims=(1, 2, 0))) * 255))
        #     image0 = cv2.applyColorMap(src=image0, colormap=cv2.COLORMAP_JET)
        #     image0 = cv2.resize(src=image0, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        #     cv2.imwrite(filename='pdm/{}.png'.format(i), img=image0)

        if self._adwf is not None:
            output = self._adwf(scores=scores)
        else:
            # outputs = []
            # for level in range(len(scores)):
            #     level_outputs = []
            #     for low in range(0, level):
            #         level_outputs.append(F.interpolate(input=scores[low], size=scores[level].shape[2:], mode='nearest'))
            #     level_outputs.append(scores[level])
            #     for high in range(level, len(scores)):
            #         level_outputs.append(F.adaptive_avg_pool2d(input=scores[high], output_size=scores[level].shape[2:]))
            #     outputs.append(F.interpolate(input=cat(tensors=level_outputs, dim=1).prod(dim=1, keepdim=True), size=self._image_size, mode='nearest'))
            # output = cat(tensors=outputs, dim=1).mean(dim=1, keepdim=True)

            scores = [F.interpolate(input=score, size=scores[0].shape[2:], mode='nearest') for score in scores]
            output = cat(tensors=scores, dim=1).prod(dim=1, keepdim=True)
        output = F.interpolate(input=output, size=self._image_size, mode='nearest')
        return output
