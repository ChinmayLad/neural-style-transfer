from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor
from PIL import Image

def loader(dataset, batch_size, num_workers=1, shuffle=True):
    data_loader = DataLoader(dataset, batch_size, num_workers=num_workers, shuffle=shuffle)
    return data_loader

class CocoDataset(Dataset):
    """
    The CocoDataset is a general dataset class used for
    loading the PIL images (JPG) and convert them to tensors.
    It does not explicitly resize any images and hence we need to make sure that
    the images are of same size prior to loading them using this class.
    If you provide the resize option the dataset will resize the smallest side to shape
    maintaining the aspent ratio and then perform center crop on the image.
    """
    def __init__(self, root, resize=None):
        super(CocoDataset, self).__init__()
        self.images = sorted(glob(root+'*.jpg'))
        self.resize = resize
        if resize:
            if len(resize) != 2:
                raise ValueError('resize tuple must contain 2 values.')
            self.transform = Compose([CenterCrop(resize), ToTensor()])
        else:
            self.transform = Compose([ToTensor()])

    def __getitem__(self, index):
        return self.get_image(self.images[index])

    def get_image(self, path):
        image = Image.open(path)
        if self.resize:
            hsize, wsize = self.resize
            w, h = image.size
            aspect_ratio = h/w
            if h < w:
                image = image.resize((int(wsize/aspect_ratio),hsize))
            else:
                image = image.resize((wsize, int(hsize*aspect_ratio)))

        image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.images)


class StyleDataset(Dataset):
    """
    The StyleDataset class is for loading style image from a directory.
    It returns the index of the style which is used as style condition
    label during traing and testing.

    The images are processed.
    In order to preserve texture and pattern data we cannot break the
    aspect-ratio of the style image. Hence the smallest size of the image
    is resized to 256px and then centercropped.
    """
    def __init__(self, root, size=256):
        super(StyleDataset, self).__init__()
        self.style_path = sorted(glob(root+'*.jpg'))
        self.size = size
        self.transform = Compose([CenterCrop((size, size)), ToTensor()])

    def __getitem__(self, index):
        return index, self.get_image(self.style_path[index])

    def get_image(self, path):
        image = Image.open(path)
        w, h = image.size
        aspect_ratio = h/w
        if h < w:
            image = image.resize((int(self.size/aspect_ratio),self.size))
        else:
            image = image.resize((self.size, int(self.size*aspect_ratio)))

        image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.style_path)

    def _get_info(self):
        return '\n'.join(["{}: {}".format(i,im) for i, im in enumerate(self.style_path)])
    
    def __repr__(self):
        return self.get_info()

    def __str__(self):
        return self.get_info()

if __name__ == "__main__":
    dataset = CocoDataset('CocoResized/')
    print(dataset.__getitem__(0).shape)
    dataset = CocoDataset('CocoResized/', resize=(256,256))
    print(dataset.__getitem__(0).shape)
    dataset = StyleDataset('StyleImages/')
    print(dataset.__getitem__(0)[1].shape)
    print(loader(dataset, 2))