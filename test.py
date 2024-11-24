from backbone.feature_extractor import Vgg19FeatureExtractor
from dataset.test_dataset import TestDataset
from ema import EMA
from evaluate import visualize
from model.adwf import ADWF
from model.faffm import FAFFM
from model.fast_pad import FastPAD
from model.fcm import FCM
from model.pam import PAM
from model.pdm import PDM
from os import listdir
from os.path import join
from torch import load
from torch.nn import ModuleList
from utils import create_dir


def run(settings: dict) -> None:
    for category in settings['categories']:
        print('========================category: {}========================'.format(category))
        category_path = join(settings['data_path'], settings['dataset'], category)
        pretrain_path = join(settings['pretrain_path'], settings['dataset'], category)
        result_path = join(settings['result_path'], settings['dataset'], category)
        create_dir(path=result_path)

        test_dataset = TestDataset(settings={
            'category_path': category_path,
            'num_workers': settings['num_workers'],
            'image_size': settings['image_size'],
            'mean': settings['mean'],
            'std': settings['std']
        })

        feature_extractor = Vgg19FeatureExtractor(path=join(settings['pretrain_path'], 'vgg19-dcbb9e9d.pth'), levels=settings['levels']).eval()
        if settings['use_pam']:
            pam = PAM(patch_size=(settings['patch_size'], settings['patch_size']), patch_stride=(settings['patch_stride'], settings['patch_stride']))
        else:
            pam = None

        if settings['check_point'] is not None:
            if settings['use_faffm']:
                faffms = ModuleList([FAFFM(channels=feature_extractor.channels, level=level, gamma=settings['gamma']) for level in range(len(feature_extractor.channels))])
                faffms.load_state_dict(state_dict=load(f=join(pretrain_path, settings['check_point'], 'faffms.pth')))
            else:
                faffms = None
            if settings['use_fcm']:
                fcms = ModuleList([FCM(num_channels=num_channels) for num_channels in feature_extractor.channels])
                fcms.load_state_dict(state_dict=load(f=join(pretrain_path, settings['check_point'], 'fcms.pth')))
            else:
                fcms = None
            pdms = ModuleList([PDM(in_channels=num_channels, betas=settings['betas']) for num_channels in feature_extractor.channels])
            pdms.load_state_dict(state_dict=load(f=join(pretrain_path, settings['check_point'], 'pdms.pth')))
            if settings['use_adwf']:
                adwf = ADWF(num_levels=len(feature_extractor.channels))
                adwf.load_state_dict(state_dict=load(f=join(pretrain_path, settings['check_point'], 'adwf.pth')))
            else:
                adwf = None
            fast_pad = FastPAD(settings={
                'feature_extractor': feature_extractor,
                'pam': pam,
                'faffms': faffms,
                'fcms': fcms,
                'pdms': pdms,
                'adwf': adwf,
                'image_size': settings['image_size'],
                'sigma': settings['sigma']
            }).to(device=settings['device'])
            ema = EMA(module=fast_pad, decay=0.99)
            ema.load(data=load(f=join(pretrain_path, settings['check_point'], 'ema.pth')))
            ema.apply_shadow()
            visualize(model=fast_pad, test_dataset=test_dataset, result_path=join(result_path, settings['check_point']))
        else:
            for pretrain_model_dir in listdir(path=pretrain_path):
                if settings['use_faffm']:
                    faffms = ModuleList([FAFFM(channels=feature_extractor.channels, level=level, gamma=settings['gamma']) for level in range(len(feature_extractor.channels))])
                    faffms.load_state_dict(state_dict=load(f=join(pretrain_path, pretrain_model_dir, 'faffms.pth')))
                else:
                    faffms = None
                if settings['use_fcm']:
                    fcms = ModuleList([FCM(num_channels=num_channels) for num_channels in feature_extractor.channels])
                    fcms.load_state_dict(state_dict=load(f=join(pretrain_path, pretrain_model_dir, 'fcms.pth')))
                else:
                    fcms = None
                pdms = ModuleList([PDM(in_channels=num_channels, betas=settings['betas']) for num_channels in feature_extractor.channels])
                pdms.load_state_dict(state_dict=load(f=join(pretrain_path, pretrain_model_dir, 'pdms.pth')))
                if settings['use_adwf']:
                    adwf = ADWF(num_levels=len(feature_extractor.channels))
                    adwf.load_state_dict(state_dict=load(f=join(pretrain_path, pretrain_model_dir, 'adwf.pth')))
                else:
                    adwf = None
                fast_pad = FastPAD(settings={
                    'feature_extractor': feature_extractor,
                    'pam': pam,
                    'faffms': faffms,
                    'fcms': fcms,
                    'pdms': pdms,
                    'adwf': adwf,
                    'image_size': settings['image_size'],
                    'sigma': settings['sigma']
                }).to(device=settings['device'])
                ema = EMA(module=fast_pad, decay=0.99)
                ema.load(data=load(f=join(pretrain_path, pretrain_model_dir, 'ema.pth')))
                ema.apply_shadow()
                visualize(model=fast_pad, test_dataset=test_dataset, result_path=join(result_path, pretrain_model_dir))
