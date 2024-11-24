from os import listdir
from os.path import join
from PIL.Image import open
from torch import zeros
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode, Normalize, Resize, ToTensor


class TestDataset(Dataset):
    def __init__(self, settings: dict) -> None:
        super(TestDataset, self).__init__()
        self._resize_transform = Resize(size=[settings['image_size'], settings['image_size']], interpolation=InterpolationMode.NEAREST)
        self._tensor_transform = ToTensor()
        self._normalize_transform = None if (settings['mean'] is None or settings['std'] is None) else Normalize(mean=settings['mean'], std=settings['std'])
        self.image_size = settings['image_size']
        self.mean = settings['mean']
        self.std = settings['std']

        self._image_paths = []
        self._image_names = []
        self._ground_truth_paths = []
        self._defect_categories = []
        image_path = join(settings['category_path'], 'test')
        ground_truth_path = join(settings['category_path'], 'ground_truth')
        for defect_category in listdir(path=image_path):
            image_defect_category_path = join(image_path, defect_category)
            ground_truth_defect_category_path = join(ground_truth_path, defect_category)
            for image_name in listdir(path=image_defect_category_path):
                self._image_paths.append(join(image_defect_category_path, image_name))
                if not defect_category == 'good':
                    self._ground_truth_paths.append(join(ground_truth_defect_category_path, image_name.split('.')[0] + '_mask.png'))
                else:
                    self._ground_truth_paths.append(None)
                self._image_names.append(image_name)
                self._defect_categories.append(defect_category)

        self.dataloader = DataLoader(dataset=self, batch_size=1, shuffle=False, num_workers=settings['num_workers'], pin_memory=True)

    def __getitem__(self, index: int) -> tuple:
        image = self._tensor_transform(pic=self._resize_transform(img=open(fp=self._image_paths[index]).convert('RGB')))
        ground_truth = zeros(size=(1, self.image_size, self.image_size)) if self._defect_categories[index] == 'good' else self._tensor_transform(pic=self._resize_transform(img=open(fp=self._ground_truth_paths[index]).convert('1')))
        return self._normalize_transform(tensor=image), ground_truth, self._defect_categories[index], self._image_names[index]

    def __len__(self) -> int:
        return len(self._image_paths)
