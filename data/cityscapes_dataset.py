import os.path
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
import glob
from data.base_dataset import BaseDataset
from models.sne_model import SNE

class cityscapesdataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.root = opt.dataroot
        # self.input_class = opt.input_class
        self.num_labels = 2
        self.use_size = (opt.useWidth, opt.useHeight) # Defined in BaseOptions
        self.is_generating_normal = True

        # if self.input_class:
        #     self.sne_model = SNE()

        if opt.phase == 'train':
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'normal_small', 'train', '*', '*.npy')))
        # elif opt.phase == 'val':
        #     self.image_list = sorted(glob.glob(os.path.join(self.root, 'normal_small', 'val', '*', '*.npy')))
        else:
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'normal_small', 'val', '*', '*.npy')))

    def __getitem__(self, index):
        city = self.image_list[index].split('/')[-2]
        normal_name = self.image_list[index].split('/')[-1] # Name of normal image
        label_name = normal_name.replace('normal.npy', 'gtFine_labelIds.png')
        rgb_name = normal_name.replace('normal.npy', 'leftImg8bit.png')

        if self.opt.phase == 'test':
            self.opt.phase = 'val'

        rgb_path = os.path.join(self.root, 'leftImg8bit', self.opt.phase, city, rgb_name)
        label_path = os.path.join(self.root, 'gtFine', self.opt.phase, city, label_name)
        normal_path = os.path.join(self.root, 'normal', self.opt.phase, city, normal_name)

        rgb_image = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        oriHeight, oriWidth, _ = rgb_image.shape

        label_image = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
        label[label_image==7] = 1
        # label[label_image==8] = 1

        normal_image = np.load(normal_path)

        rgb_image = cv2.resize(rgb_image, self.use_size)
        label = cv2.resize(label, self.use_size, interpolation=cv2.INTER_NEAREST)
        normal_image = cv2.resize(normal_image, self.use_size)

        rgb_image = rgb_image.astype(np.float32) / 255
        rgb_image = transforms.ToTensor()(rgb_image)

        # convert shape from [h, w, 3] to [3, h, w]
        another_image = transforms.ToTensor()(normal_image)

        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)

        # return a dictionary containing useful information
        # input rgb images, another images and labels for training;
        # 'path': image name for saving predictions
        # 'oriSize': original image size for evaluating and saving predictions
        return {'rgb_image': rgb_image, 'another_image': another_image, 'label': label,
                'path': self.opt.name, 'oriSize': (oriWidth, oriHeight)}

    def __len__(self):
        return len(self.image_list)

    def name(self):
        return 'cityscapes'
