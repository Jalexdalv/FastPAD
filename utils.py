import random
import numpy.random
from cv2 import cvtColor, COLOR_RGB2BGR
from numpy import around, ndarray, uint8
from os import environ, makedirs
from os.path import exists
from torch import cuda, device, from_numpy, manual_seed, Tensor
from torch.backends import cudnn
from torch.nn import LayerNorm, Linear, Module, ReLU, Sequential


def set_seed(seed: int) -> None:
    environ['PYTHONHASHSEED'] = str(seed)
    random.seed(a=seed)
    numpy.random.seed(seed=seed)
    manual_seed(seed=seed)
    cuda.manual_seed(seed=seed)
    cuda.manual_seed_all(seed=seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def create_dir(path: str):
    if not exists(path):
        makedirs(name=path)


def get_device(module: Module) -> device:
    return next(module.parameters()).device


def convert_tensor_to_numpy(tensor: Tensor) -> ndarray:
    return tensor.detach().cpu().numpy()


def convert_numpy_to_tensor(ndarray: ndarray, dev: device) -> Tensor:
    return from_numpy(ndarray).to(device=dev)


def convert_tensor_to_opencv_image(image: Tensor, mean, std) -> ndarray:
    image = unormalize_image(image=convert_tensor_to_numpy(tensor=image), mean=mean, std=std, opencv=False)
    if len(image.shape) == 3:
        image = image.transpose(1, 2, 0)
        image = cvtColor(src=image, code=COLOR_RGB2BGR)
    return image


def unormalize_image(image: ndarray, mean, std, opencv: bool = False) -> ndarray:
    if len(image.shape) == 2:
        image = image * std + mean
    elif opencv:
        if image.shape[2] == 1:
            image[:, :, 0] = image[:, :, 0] * std + mean
        elif image.shape[2] == 3:
            if isinstance(mean, int) or isinstance(mean, float):
                mean = [mean, mean, mean]
            if isinstance(std, int) or isinstance(std, float):
                std = [std, std, std]
            image[:, :, 0] = image[:, :, 0] * std[2] + mean[2]
            image[:, :, 1] = image[:, :, 1] * std[1] + mean[1]
            image[:, :, 2] = image[:, :, 2] * std[0] + mean[0]
    elif not opencv:
        if image.shape[0] == 1:
            image[0] = image[0] * std + mean
        elif image.shape[0] == 3:
            if isinstance(mean, int) or isinstance(mean, float):
                mean = [mean, mean, mean]
            if isinstance(std, int) or isinstance(std, float):
                std = [std, std, std]
            image[0] = image[0] * std[0] + mean[0]
            image[1] = image[1] * std[1] + mean[1]
            image[2] = image[2] * std[2] + mean[2]
    return uint8(around(a=image * 255))


def max_min_normalize(input: ndarray) -> ndarray:
    input_max = input.max()
    input_min = input.min()
    output = (input - input_min) / (input_max - input_min)
    return output


def get_mlp_block(in_features: int, out_features: int) -> Sequential:
    return Sequential(
        Linear(in_features=in_features, out_features=out_features, bias=False),
        LayerNorm(normalized_shape=out_features, bias=True),
        ReLU(inplace=True)
    )
