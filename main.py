import data_cleanse
import preprocess
import test
import train
from argparse import ArgumentParser
from torch import device
from utils import set_seed


if __name__ == '__main__':
    argumentParser = ArgumentParser()

    argumentParser.add_argument('--mode', type=str, choices=('data_cleanse', 'preprocess', 'train', 'test'), default='train')

    argumentParser.add_argument('--device', type=int, default=0)

    argumentParser.add_argument('--seed', type=int, default=42)

    argumentParser.add_argument('--dataset', type=str, choices=('mvtec', 'visa', 'bhaad'), default='mvtec')
    argumentParser.add_argument('--mean', nargs='+', type=float, default=(0.485, 0.456, 0.406))
    argumentParser.add_argument('--std', nargs='+', type=float, default=(0.229, 0.224, 0.225))

    argumentParser.add_argument('--image_size', type=int, default=256)

    argumentParser.add_argument('--batch_size', type=int, default=4)
    argumentParser.add_argument('--num_workers', type=int, default=4)
    argumentParser.add_argument('--num_epochs', type=int, default=500)
    argumentParser.add_argument('--lr', type=float, default=1e-5)
    argumentParser.add_argument('--weight_decay', type=float, default=5e-5)

    argumentParser.add_argument('--patch_size', type=int, default=4)
    argumentParser.add_argument('--patch_stride', type=int, default=4)
    argumentParser.add_argument('--betas', nargs='+', type=int, default=(2, 1, 2, 1, 2, 1))
    argumentParser.add_argument('--gamma', type=int, default=16)
    argumentParser.add_argument('--sigma', type=float, default=4.)

    argumentParser.add_argument('--levels', nargs='+', type=str, default=('level_2', 'level_3', 'level_4', 'level_5'))

    argumentParser.add_argument('--use_sam', type=int, choices=(0, 1), default=1)
    argumentParser.add_argument('--use_ascm', type=int, choices=(0, 1), default=1)
    argumentParser.add_argument('--use_pam', type=int, choices=(0, 1), default=1)
    argumentParser.add_argument('--use_faffm', type=int, choices=(0, 1), default=1)
    argumentParser.add_argument('--use_fcm', type=int, choices=(0, 1), default=1)
    argumentParser.add_argument('--use_adwf', type=int, choices=(0, 1), default=1)

    # MVTEC
    # argumentParser.add_argument('--categories', nargs='+', type=str, default=('carpet', 'grid', 'leather', 'wood', 'tile', 'bottle', 'cable'))
    # argumentParser.add_argument('--backgrounds', nargs='+', type=str, choices=('white', 'black'), default=('none', 'none', 'none', 'none', 'none', 'white', 'black', 'white', 'black', 'black', 'black', 'white', 'black', 'white', 'white'))

    # argumentParser.add_argument('--categories', nargs='+', type=str, default=('capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper'))
    # argumentParser.add_argument('--backgrounds', nargs='+', type=str, choices=('white', 'black'), default=('none', 'none', 'none', 'none', 'none', 'white', 'black', 'white', 'black', 'black', 'black', 'white', 'black', 'white', 'white'))

    argumentParser.add_argument('--categories', nargs='+', type=str, default=('hazelnut',))
    argumentParser.add_argument('--backgrounds', nargs='+', type=str, choices=('white', 'black'), default=('none', 'none', 'none', 'none', 'none', 'white', 'black', 'white', 'black', 'black', 'black', 'white', 'black', 'white', 'white'))

    # VISA
    # argumentParser.add_argument('--categories', nargs='+', type=str, default=('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'))
    # argumentParser.add_argument('--backgrounds', nargs='+', type=str, choices=('none', 'white', 'black'), default=('black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black'))

    # BHAAD
    # argumentParser.add_argument('--categories', nargs='+', type=str, default=('pcb',))
    # argumentParser.add_argument('--backgrounds', nargs='+', type=str, choices=('none', 'white', 'black'), default=('black',))

    argumentParser.add_argument('--check_point', type=str, default=None)

    argumentParser.add_argument('--data_path', type=str, default='../autodl-tmp/data')
    argumentParser.add_argument('--pretrain_path', type=str, default='../autodl-tmp/pretrain')
    argumentParser.add_argument('--result_path', type=str, default='result')

    argumentParser.add_argument('--timing', type=int, choices=(0, 1), default=0)
    argumentParser.add_argument('--evaluate_interval', type=int, default=10)
    args = argumentParser.parse_args()

    if args.seed is not None:
        set_seed(seed=args.seed)
        print('set seed: {}'.format(args.seed))

    settings = {
        'device': device('cpu') if args.device == -1 else device('cuda:{}'.format(args.device)),
        'dataset': args.dataset,
        'mean': args.mean,
        'std': args.std,
        'image_size': args.image_size,
        'num_workers': args.num_workers,
        'patch_size': args.patch_size,
        'patch_stride': args.patch_stride,
        'betas': args.betas,
        'gamma': args.gamma,
        'sigma': args.sigma,
        'levels': args.levels,
        'use_ascm': args.use_ascm,
        'use_pam': args.use_pam,
        'use_faffm': args.use_faffm,
        'use_fcm': args.use_fcm,
        'use_adwf': args.use_adwf,
        'categories': args.categories,
        'check_point': args.check_point,
        'data_path': args.data_path,
        'pretrain_path': args.pretrain_path,
        'timing': args.timing
    }
    if args.mode == 'data_cleanse':
        data_cleanse.run(settings=settings)
    elif args.mode == 'preprocess':
        settings['use_sam'] = args.use_sam
        settings['backgrounds'] = args.backgrounds
        preprocess.run(settings=settings)
    elif args.mode == 'train':
        settings['batch_size'] = args.batch_size
        settings['num_epochs'] = args.num_epochs
        settings['lr'] = args.lr
        settings['weight_decay'] = args.weight_decay
        settings['evaluate_interval'] = args.evaluate_interval
        train.run(settings=settings)
    elif args.mode == 'test':
        settings['result_path'] = args.result_path
        test.run(settings=settings)
