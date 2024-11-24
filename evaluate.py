from cv2 import applyColorMap, COLORMAP_JET, imwrite
from dataset.test_dataset import TestDataset
from fast_auc import fast_numba_auc
from numpy import array
from os.path import join
from time import time
from torch import no_grad
from torch.nn import Module
from utils import convert_tensor_to_numpy, convert_tensor_to_opencv_image, create_dir, get_device, max_min_normalize, unormalize_image


def _binary_score(model: Module, test_dataset: TestDataset, timing: bool) -> tuple:
    scores, ground_truths = [], []
    for input, ground_truth, _, _ in test_dataset.dataloader:
        if timing:
            time_start = time()
        score = model.compute_score(input=input.to(device=get_device(module=model)))
        if timing:
            time_end = time()
            print(1. / (time_end - time_start), 'FPS')
        scores.append(score)
        ground_truths.append(convert_tensor_to_numpy(tensor=ground_truth.squeeze()))
    scores = array(object=scores)
    ground_truths = array(object=ground_truths).astype(bool)
    return scores, ground_truths


def compute_auc_roc(model: Module, test_dataset: TestDataset, timing: bool) -> float:
    with no_grad():
        scores, ground_truths = _binary_score(model=model, test_dataset=test_dataset, timing=timing)
        auc_roc = fast_numba_auc(y_true=ground_truths.ravel(), y_score=scores.ravel())
        print('auc-roc: {}'.format(auc_roc))
    # pro = compute_pro(ground_truths, scores)
    # print('auc-pro：{}'.format(pro))
    return auc_roc


# import pandas as pd
# from skimage import measure
# import numpy as np
# import cv2
# from sklearn import metrics
# def compute_pro(masks, amaps, num_th=20):
#
#     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
#     binary_amaps = np.zeros_like(amaps, dtype=bool)
#
#     min_th = amaps.min()
#     max_th = amaps.max()
#     delta = (max_th - min_th) / num_th
#
#     k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     for th in np.arange(min_th, max_th, delta):
#         binary_amaps[amaps <= th] = 0
#         binary_amaps[amaps > th] = 1
#
#         pros = []
#         for binary_amap, mask in zip(binary_amaps, masks):
#             binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
#             for region in measure.regionprops(measure.label(mask)):
#                 axes0_ids = region.coords[:, 0]
#                 axes1_ids = region.coords[:, 1]
#                 tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
#                 pros.append(tp_pixels / region.area)
#
#         inverse_masks = 1 - masks
#         fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
#         fpr = fp_pixels / inverse_masks.sum()
#
#         df = df._append({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
#
#     # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
#     df = df[df["fpr"] < 0.3]
#     df["fpr"] = df["fpr"] / df["fpr"].max()
#
#     pro_auc = metrics.auc(df["fpr"], df["pro"])
#     return pro_auc


def visualize(model: Module, test_dataset: TestDataset, result_path: str) -> None:
    with no_grad():
        for input, ground_truth, defect_category, name in test_dataset.dataloader:
            score = model.compute_score(input=input.to(device=get_device(module=model)))
            score = max_min_normalize(input=score)
            input = convert_tensor_to_opencv_image(image=input.squeeze(), mean=test_dataset.mean, std=test_dataset.std)
            ground_truth = convert_tensor_to_opencv_image(image=ground_truth.squeeze(), mean=0., std=1.)
            path = join(result_path, defect_category[0], name[0])
            create_dir(path=path)
            imwrite(filename=join(path, 'ground_truth.png'), img=ground_truth)
            input = input[:, :, (2, 1, 0)]
            heat_map = applyColorMap(src=unormalize_image(image=score, mean=0., std=1.), colormap=COLORMAP_JET) * 0.5 + input * 0.7
            heat_map_path = join(path, 'heat_map.png')
            imwrite(filename=heat_map_path, img=heat_map)
            print("success output result {}".format(path))
