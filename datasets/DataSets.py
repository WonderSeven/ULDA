import io
import torch.utils.data as data
from PIL import Image
from skimage import io

__all__ = ['DataSet']


def default_loader(path):
    return io.imread(path)


def RGB_loader(path):
    return Image.open(path).convert('RGB')


class DataSet(data.Dataset):
    def __init__(self, imgNames, transform=None, loader=RGB_loader):
        self.imgs = imgNames
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

