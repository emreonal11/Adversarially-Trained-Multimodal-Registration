import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch


def random_crop(A, B, crop_size):  # randomize a crop given two tensor images A and B
    rand_x = int(random.random() * (A.shape[2] - crop_size))
    rand_y = int(random.random() * (A.shape[1] - crop_size))
    return A[:, rand_y:rand_y+crop_size, rand_x:rand_x+crop_size], B[:, rand_y:rand_y+crop_size, rand_x:rand_x+crop_size]


def random_flip(A, B, prob):  # randomize a horizontal flip given two tensor images A and B
    if random.random() < prob:
        return torch.flip(A, [2]), torch.flip(B, [2])
    return A, B


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        assert(self.A_size == self.B_size) # assert the two modalities have same num_images (should be the case for paired images)
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        A_path = self.A_paths[index] 
        B_path = self.B_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        # assert images are paired, then apply image transformation
        assert(A_path.split('/')[-1] == B_path.split('/')[-1]) # remove assertion if each img in pair doesnt have same filename

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        # perform random crop
        if 'crop' in self.opt.preprocess:
            A, B = random_crop(A, B, self.opt.crop_size)

        # perform random flip
        if not self.opt.no_flip:
            A, B = random_flip(A, B, 0.5)


        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return min(self.A_size, self.B_size)

