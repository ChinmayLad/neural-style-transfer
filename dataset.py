from glob import glob
from torchvision.transforms import Compose, CenterCrop, ToTensor


def loader(dataset, batch_size, num_workers=1, shuffle=True):
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size,
                                              num_workers=num_workers,
                                              shuffle=shuffle)
    return data_loader

class CustomDataset(Dataset):
    def __init__(self, root):
        super(CustomDataset, self).__init__()
        self.images = sorted(glob(root+'*.jpg'))
        self.transform = Compose([ToTensor()])

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.images)


class StyleDataset(Dataset):
    def __init__(self, root, size=256):
        super(StyleDataset, self).__init__()
        self.style_path = sorted(glob(root+'*.jpg'))
        self.size = size
        self.transform = Compose([CenterCrop((size, size)), ToTensor()])

    def __getitem__(self, index):
        image = Image.open(self.style_path[index])
        w, h = image.size
        aspect_ratio = h/w
        if h < w:
            image = image.resize((int(self.size/aspect_ratio),self.size))
        else:
            image = image.resize((self.size, int(self.size*aspect_ratio)))

        image = self.transform(image)
        return index, image
    
    def __len__(self):
        return len(self.style_path)

    def _get_info(self):
        return '\n'.join(["{}: {}".format(i,im) for i, im in enumerate(self.style_path)])
    
    def __repr__(self):
        return self.get_info()

    def __str__(self):
        return self.get_info()
