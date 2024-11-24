from cv2 import imread, imwrite
from os import listdir, remove
from os.path import exists, join
from shutil import rmtree
from utils import create_dir, unormalize_image


def run(settings: dict) -> None:
    # VisA
    visa_path = join(settings['data_path'], 'visa')
    if exists(path=join(visa_path, 'split_csv')):
        rmtree(path=join(visa_path, 'split_csv'))
    if exists(path=join(visa_path, 'LICENSE-DATASET')):
        remove(path=join(visa_path, 'LICENSE-DATASET'))
    if exists(path=join(visa_path, 'meta.json')):
        remove(path=join(visa_path, 'meta.json'))
    for category in listdir(path=visa_path):
        category_path = join(visa_path, category)
        train_path = join(category_path, 'Data', 'Images', 'Normal')
        new_train_path = join(category_path, 'train', 'good')
        test_path = join(category_path, 'Data', 'Images', 'Anomaly')
        new_test_path = join(category_path, 'test', 'anomaly')
        ground_truth_path = join(category_path, 'Data', 'Masks', 'Anomaly')
        new_ground_truth_path = join(category_path, 'ground_truth', 'anomaly')
        prior_test_path = join(category_path, 'Data', 'Priors', 'Images')
        new_prior_test_path = join(category_path, 'prior', 'image')
        prior_ground_truth_path = join(category_path, 'Data', 'Priors', 'Masks')
        new_prior_ground_truth_path = join(category_path, 'prior', 'ground_truth')
        create_dir(path=new_train_path)
        create_dir(path=new_test_path)
        create_dir(path=new_ground_truth_path)
        create_dir(path=new_prior_test_path)
        create_dir(path=new_prior_ground_truth_path)
        if exists(train_path):
            for image_name in listdir(path=train_path):
                image = imread(filename=join(train_path, image_name))
                imwrite(filename=join(new_train_path, image_name.split('.')[0] + '.png'), img=image)
                print('succeed cleanse train {} {}'.format(category, image_name))
        if exists(test_path):
            for image_name in listdir(path=test_path):
                image = imread(filename=join(test_path, image_name))
                if int(image_name.split('.')[0]) == 0 or int(image_name.split('.')[0]) == 1:
                    imwrite(filename=join(new_prior_test_path, image_name.split('.')[0] + '.png'), img=image)
                    print('succeed cleanse prior test {} {}'.format(category, image_name))
                imwrite(filename=join(new_test_path, image_name.split('.')[0] + '.png'), img=image)
                print('succeed cleanse test {} {}'.format(category, image_name))
        if exists(ground_truth_path):
            for image_name in listdir(path=ground_truth_path):
                image = imread(filename=join(ground_truth_path, image_name))
                image[image > 1] = 1
                image[image < 1] = 0
                image = unormalize_image(image=image, mean=0., std=1., opencv=True)
                if int(image_name.split('.')[0]) == 0 or int(image_name.split('.')[0]) == 1:
                    imwrite(filename=join(new_prior_ground_truth_path, image_name.split('.')[0] + '_mask.png'), img=image)
                    print('succeed cleanse prior ground_truth {} {}'.format(category, image_name))
                imwrite(filename=join(new_ground_truth_path, image_name.split('.')[0] + '_mask.png'), img=image)
                print('succeed cleanse ground_truth {} {}'.format(category, image_name))
        if exists(path=join(category_path, 'image_anno.csv')):
            remove(path=join(category_path, 'image_anno.csv'))
        if exists(path=train_path):
            rmtree(path=train_path)
        if exists(path=test_path):
            rmtree(path=test_path)
        if exists(path=ground_truth_path):
            rmtree(path=ground_truth_path)
        if exists(path=prior_test_path):
            rmtree(path=prior_test_path)
        if exists(path=prior_ground_truth_path):
            rmtree(path=prior_ground_truth_path)
        if exists(join(category_path, 'Data')):
            rmtree(path=join(category_path, 'Data'))
