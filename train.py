from backbone.feature_extractor import Vgg19FeatureExtractor
from dataset.test_dataset import TestDataset
from dataset.train_dataset import TrainDataset
from ema import EMA
from evaluate import compute_auc_roc
from model.adwf import ADWF
from model.faffm import FAFFM
from model.fast_pad import FastPAD
from model.fcm import FCM
from model.pam import PAM
from model.pdm import PDM
from os.path import join
from torch import load, save
from torch.nn import ModuleList
from torch.optim import AdamW
from tqdm import tqdm
from utils import create_dir


def run(settings: dict) -> None:
    mask_path = join(settings['data_path'], 'mask')
    for category in settings['categories']:
        print('========================category: {}========================'.format(category))
        category_path = join(settings['data_path'], settings['dataset'], category)
        pretrain_path = join(settings['pretrain_path'], settings['dataset'], category)
        create_dir(path=pretrain_path)

        train_dataset = TrainDataset(settings={
            'use_ascm': settings['use_ascm'],
            'mask_path': mask_path,
            'category_path': category_path,
            'batch_size': settings['batch_size'],
            'num_workers': settings['num_workers'],
            'image_size': settings['image_size'],
            'mean': settings['mean'],
            'std': settings['std']
        })
        test_dataset = TestDataset(settings={
            'category_path': category_path,
            'num_workers': settings['num_workers'],
            'image_size': settings['image_size'],
            'mean': train_dataset.mean,
            'std': train_dataset.std
        })

        feature_extractor = Vgg19FeatureExtractor(path=join(settings['pretrain_path'], 'vgg19-dcbb9e9d.pth'), levels=settings['levels']).eval()
        if settings['use_pam']:
            pam = PAM(patch_size=(settings['patch_size'], settings['patch_size']), patch_stride=(settings['patch_stride'], settings['patch_stride']))
        else:
            pam = None
        if settings['use_faffm']:
            faffms = ModuleList([FAFFM(channels=feature_extractor.channels, level=level, gamma=settings['gamma']) for level in range(len(feature_extractor.channels))])
            if settings['check_point'] is not None:
                faffms.load_state_dict(state_dict=load(f=join(pretrain_path, settings['check_point'], 'faffms.pth')))
        else:
            faffms = None
        if settings['use_fcm']:
            fcms = ModuleList([FCM(num_channels=num_channels) for num_channels in feature_extractor.channels])
            if settings['check_point'] is not None:
                fcms.load_state_dict(state_dict=load(f=join(pretrain_path, settings['check_point'], 'fcms.pth')))
        else:
            fcms = None
        pdms = ModuleList([PDM(in_channels=num_channels, betas=settings['betas']) for num_channels in feature_extractor.channels])
        if settings['check_point'] is not None:
            pdms.load_state_dict(state_dict=load(f=join(pretrain_path, settings['check_point'], 'pdms.pth')))
        if settings['use_adwf']:
            adwf = ADWF(num_levels=len(feature_extractor.channels))
            if settings['check_point'] is not None:
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

        print('============training============')
        optimizer = AdamW(params=fast_pad.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
        ema = EMA(module=fast_pad, decay=0.99)
        ema.register()
        for index in range(1, settings['num_epochs'] + 1):
            print('epoch: {}'.format(index))
            loss_sum, num_samples, auc_roc = 0, 0, 0
            with tqdm(iterable=train_dataset.dataloader, unit='batch') as batches:
                for input, augmented_input, mask in batches:
                    loss = fast_pad.compute_loss(input=input.to(device=settings['device']), augmented_input=augmented_input.to(device=settings['device']), mask=mask.to(device=settings['device']))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ema.update()
                    loss_sum += loss.item()
                    num_samples += input.shape[0]
            print('loss: {}'.format(loss_sum / num_samples))

            if index % settings['evaluate_interval'] == 0:
                ema.apply_shadow()
                auc_roc = compute_auc_roc(model=fast_pad, test_dataset=test_dataset, timing=settings['timing'])
                ema.restore()
                model_path = join(pretrain_path, str(auc_roc) + '-{}'.format(index))
                create_dir(path=model_path)
                if settings['use_faffm']:
                    save(obj=faffms.state_dict(), f=join(model_path, 'faffms.pth'))
                if settings['use_fcm']:
                    save(obj=fcms.state_dict(), f=join(model_path, 'fcms.pth'))
                if settings['use_adwf']:
                    save(obj=adwf.state_dict(), f=join(model_path, 'adwf.pth'))
                save(obj=pdms.state_dict(), f=join(model_path, 'pdms.pth'))
                save(obj=ema.save(), f=join(model_path, 'ema.pth'))
                print('saving completed')
