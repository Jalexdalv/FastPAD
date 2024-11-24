from os import listdir
from os.path import exists, join
from PIL.Image import open
from torch import cat, randint, randn_like, randperm, stack
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode, Normalize, Resize, ToTensor


class TrainDataset(Dataset):
    def __init__(self, settings: dict) -> None:
        super(TrainDataset, self).__init__()
        self._use_ascm = settings['use_ascm']
        self._resize_transform = Resize(size=[settings['image_size'], settings['image_size']], interpolation=InterpolationMode.NEAREST)
        self._tensor_transform = ToTensor()
        self._normalize_transform = None if (settings['mean'] is None or settings['std'] is None) else Normalize(mean=settings['mean'], std=settings['std'])
        self.image_size = settings['image_size']
        self.mean = settings['mean']
        self.std = settings['std']

        self._prior_pixels = []
        prior_path = join(settings['category_path'], 'prior')
        prior_image_path = join(prior_path, 'image')
        prior_ground_truth_path = join(prior_path, 'ground_truth')
        if exists(prior_image_path) and exists(prior_ground_truth_path):
            for image_name in listdir(path=prior_image_path):
                image = self._tensor_transform(pic=self._resize_transform(img=open(fp=join(prior_image_path, image_name)).convert('RGB')))
                ground_truth = self._tensor_transform(pic=self._resize_transform(img=open(fp=join(prior_ground_truth_path, image_name.split('.')[0] + '_mask.png')).convert('1')))
                defect_area = ground_truth[0] == 1
                self._prior_pixels.append(stack(tensors=[image[0][defect_area], image[1][defect_area], image[2][defect_area]], dim=0))
            self._prior_pixels = cat(tensors=self._prior_pixels, dim=1)
            self._prior_pixels = self._prior_pixels[:, randperm(n=self._prior_pixels.shape[1])]

        self._image_paths = []
        self._foreground_paths = []
        self._mask_paths = []
        image_dir_path = join(settings['category_path'], 'train', 'good')
        foreground_dir_path = join(settings['category_path'], 'foreground')
        for image_name in listdir(path=image_dir_path):
            self._image_paths.append(join(image_dir_path, image_name))
            if exists(foreground_dir_path):
                self._foreground_paths.append(join(foreground_dir_path, image_name))
            else:
                self._foreground_paths.append(None)
        for mask_name in listdir(path=settings['mask_path']):
            self._mask_paths.append(join(settings['mask_path'], mask_name))
        self._image_cache = {}
        self._foreground_cache = {}
        self._mask_cache = {}

        self.dataloader = DataLoader(dataset=self, batch_size=settings['batch_size'], shuffle=True, num_workers=settings['num_workers'], drop_last=True, pin_memory=True)

    def __getitem__(self, index: int) -> tuple:
        mask_index = randint(low=0, high=len(self._mask_paths), size=(1,)).item()
        if index in self._mask_cache:
            mask = self._mask_cache[index]
        else:
            mask = self._tensor_transform(pic=self._resize_transform(img=open(fp=self._mask_paths[mask_index]).convert('1')))
            self._mask_cache[mask_index] = mask
        if index in self._image_cache:
            image = self._image_cache[index]
        else:
            image = self._tensor_transform(pic=self._resize_transform(img=open(fp=self._image_paths[index]).convert('RGB')))
            self._image_cache[index] = image
        if self._use_ascm:
            if self._foreground_paths[index] is not None:
                if index in self._foreground_cache:
                    foreground = self._foreground_cache[index]
                else:
                    foreground = self._tensor_transform(pic=self._resize_transform(img=open(fp=self._foreground_paths[index]).convert('1')))
                    self._foreground_cache[index] = foreground
                mask = mask * foreground
            augmented_image = image.clone()
            augmented_area = mask[0] == 1
            random_select_index = randint(low=0, high=self._prior_pixels.shape[1], size=(augmented_area.sum().item(),))
            augmented_image[0][augmented_area] = self._prior_pixels[0][random_select_index]
            augmented_image[1][augmented_area] = self._prior_pixels[1][random_select_index]
            augmented_image[2][augmented_area] = self._prior_pixels[2][random_select_index]
        else:
            augmented_image = randn_like(input=image)
            normal_area = mask[0] == 0
            augmented_image[0][image] = image[0][normal_area]
            augmented_image[1][image] = image[1][normal_area]
            augmented_image[2][image] = image[2][normal_area]
        return self._normalize_transform(tensor=image), self._normalize_transform(tensor=augmented_image), mask

    def __len__(self) -> int:
        return len(self._image_paths)
