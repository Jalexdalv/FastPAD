from cv2 import bitwise_not, COLOR_BGR2RGB, connectedComponentsWithStats, cvtColor, erode, imread, IMREAD_GRAYSCALE, IMREAD_COLOR, imwrite, threshold, THRESH_BINARY, THRESH_OTSU
from numpy import array, ones, zeros
from os import listdir
from os.path import join
from segment_anything import SamPredictor, sam_model_registry
from utils import create_dir, unormalize_image


def run(settings: dict) -> None:
    sam_model = sam_model_registry['vit_h'](checkpoint=join(settings['pretrain_path'], 'sam_vit_h_4b8939.pth')).to(device=settings['device']).eval()
    sam_predictor = SamPredictor(sam_model=sam_model)

    dataset_path = join(settings['data_path'], settings['dataset'])
    for category, background in zip(settings['categories'], settings['backgrounds']):
        category_path = join(dataset_path, category)
        train_path = join(category_path, 'train', 'good')
        foreground_path = join(category_path, 'foreground')
        for image_name in listdir(path=train_path):
            if background != 'none':
                if settings['use_sam']:
                    image = cvtColor(src=imread(filename=join(train_path, image_name), flags=IMREAD_COLOR), code=COLOR_BGR2RGB)
                    sam_predictor.set_image(image=image)
                    point_coords = array(object=[[1, 1], [image.shape[0] - 1, image.shape[1] - 1]])
                    point_labels = array(object=[1, 1])
                    background_mask = sam_predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=False)[0][0]
                    foreground_mask = ~background_mask
                    foreground_mask = unormalize_image(image=foreground_mask, mean=0., std=1., opencv=False)
                else:
                    image = imread(filename=join(train_path, image_name), flags=IMREAD_GRAYSCALE)
                    foreground_mask = threshold(src=image, thresh=0, maxval=255, type=THRESH_BINARY + THRESH_OTSU)[1]
                    if background == 'white':
                        foreground_mask = bitwise_not(src=foreground_mask)
                    foreground_mask = erode(src=foreground_mask, kernel=ones(shape=(5, 5)))
                if settings['dataset'] == 'mvtec' and category == 'transistor':
                    areas = []
                    for stat in connectedComponentsWithStats(image=foreground_mask, connectivity=8)[2]:
                        if stat[3] != foreground_mask.shape[0] or stat[2] != foreground_mask.shape[1]:
                            areas.append(stat)
                    areas.sort(key=lambda area: area[-1], reverse=True)
                    for index in range(1, len(areas)):
                        x, y, w, h, s = areas[index]
                        foreground_mask[y: y + h, x: x + w] = zeros(shape=(h, w))
                create_dir(path=foreground_path)
                imwrite(filename=join(foreground_path, image_name), img=foreground_mask)
                print('succeed segment foreground {} {}'.format(category, image_name))
