import os
import glob
import PIL.Image
import random
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from typing import Tuple, Optional

import utils

class RandomImagePixelationDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        width_range: tuple[int, int],
        height_range: tuple[int, int],
        size_range: tuple[int, int],
        dtype: Optional[type] = None
    ):
        """
        Provide pixelated images and their additional data.
        Args:
            image_dir: directory where image files with the extension ".jpg" should be searched for
            width_range, height_range, size_range: 2-tuples (minimum, maximum), range from which a random sample will be chosen
            dtype: the data type of the loaded images
        """
        for min_value, max_value in [width_range, height_range, size_range]:
            if min_value < 2:
                raise ValueError("The minimum value can't be smaller than 2.")
            if min_value > max_value:
                raise ValueError("The minimum value can't be greater than the maximum value.")
        
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        self.dtype = dtype

        if not os.path.isdir(image_dir):
            raise ValueError(f"Directory {image_dir} does not exist.")
        
        # search in folder "image_dir" and in all its subdirectories for files ending with ".jpg":
        found_files = glob.glob(os.path.join(os.path.abspath(image_dir), "**", "*.jpg"), recursive=True)
        found_files.sort()
        self.found_files = found_files


    def __getitem__(self, index: int):
        # reshape so that the images have the same shape as in the test set
        im_shape = 64
        resize_transforms = transforms.Compose([
            transforms.Resize(size=im_shape, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size=(im_shape, im_shape)),
            ])
        with PIL.Image.open(self.found_files[index]) as image:
            image = resize_transforms(image)

        # transform to grayscale
        array_image = np.array(image, dtype=self.dtype)
        grayscale_image = utils.to_grayscale(array_image)

        # generate numbers for prepare_image function 
        image_width, image_height = image.size
        rng = np.random.default_rng(index)
        width = rng.integers(low=self.width_range[0], high=self.width_range[1], endpoint=True)
        width = np.clip(width, a_min=0, a_max=image_width)
        height = rng.integers(low=self.height_range[0], high=self.height_range[1], endpoint=True)
        height = np.clip(height, a_min=0, a_max=image_height)
        x = rng.integers(low=0, high=image_width-width, endpoint=True)
        y = rng.integers(low=0, high=image_height-height, endpoint=True)
        size = rng.integers(low=self.size_range[0], high=self.size_range[1], endpoint=True)

        original_image, pixelated_image, known_array = utils.prepare_image(grayscale_image, x, y, width, height, size)
        concat_pixelated_known = np.concatenate((pixelated_image, known_array), axis=0)

        return concat_pixelated_known, original_image
    

    def __len__(self):
        return len(self.found_files)
    
    

def create_concat_dataset(folders: list[str]) -> Dataset:
    # folders: list of absolute paths to the folders with images
    # return torch dataset containing images from all folders
    dataset_list = []
    for folder in folders:
        dataset_list.append(
            RandomImagePixelationDataset(
                image_dir=folder,
                width_range=(4, 32),
                height_range=(4, 32),
                size_range=(4, 16),
                dtype=np.float32
            ))
    dataset = ConcatDataset(dataset_list)
    return dataset


def create_train_val_datasets(image_dir: str, split_perc: int = 0.8, random_seed: int = 0) -> Tuple[Dataset, Dataset]:
    # list of absolute paths of all files in the image_dir directory
    found_dirs = glob.glob(os.path.join(os.path.abspath(image_dir), "**"))
    # shuffle found_dirs list
    random.seed(random_seed)
    random.shuffle(found_dirs)
    # split into training and validation part
    train_val_idx = int(split_perc * len(found_dirs))
    train_dirs = found_dirs[:train_val_idx]
    valid_dirs = found_dirs[train_val_idx:]
    # create datasets
    train_set = create_concat_dataset(train_dirs)
    valid_set = create_concat_dataset(valid_dirs)
    return train_set, valid_set
