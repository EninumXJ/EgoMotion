import argparse
import os
import torch
from util import utils

class TestOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default='test01', help='experiment name')
        parser.add_argument('--dataroot', default='/data/newhome/litianyi/dataset/EgoMotion/', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--config', default='/home/litianyi/workspace/EgoMotion/remy_2scenes.yml', help='path to config of EgoMotion dataset')
        parser.add_argument('--gpus', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--resume', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--features_dir', type=str, default='/home/litianyi/workspace/EgoMotion/model/features/', help='features are saved here')
        parser.add_argument('--log_dir', type=str, default='/home/litianyi/workspace/EgoMotion/logs/tensorboard/', help='models are saved here')
        # model parameters
        # parser.add_argument('--model', type=str, default='pix2pix', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--arch', type=str, default='r3d', choices=['r3d','unet_256'])
        parser.add_argument('--clip_length', type=int, default=10, help='length of clip as network input')
        # dataset parameters
        parser.add_argument('--image_tmpl', type=str, default='{:04d}.jpg', help='templates of image names in dataset')
        parser.add_argument('--val_size', type=int, default=1000, help='size of val set.')
        parser.add_argument('--batch_size', type=int, default=1, help='size of a mini batch.')
        # training parameters
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--use_feature', action='store_true')
        parser.add_argument('--feature_path', default='', type=str,)
        self.isTrain = True
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()
    
    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpus.split(',')
        opt.gpus = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpus.append(id)
        if len(opt.gpus) > 0:
            torch.cuda.set_device(opt.gpus[0])

        self.opt = opt
        return self.opt
    
    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.log_dir, opt.name)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')