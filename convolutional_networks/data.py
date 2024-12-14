import numpy as np
import torch
import torchvision
import os

import imageio
import os

from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image

from tqdm import tqdm


from fake import FakeData
from fourier_slicer import FourierSlicer


__all__ = ['loaders']

c10_classes = np.array([[0, 1, 2, 8, 9], [3, 4, 5, 6, 7]], dtype=np.int32)

def svhn_loaders(
    path,
    batch_size,
    num_workers,
    transform_train,
    transform_test,
    use_validation,
    val_size,
    shuffle_train=True,
):
    train_set = torchvision.datasets.SVHN(
        root=path, split="train", download=True, transform=transform_train
    )

    if use_validation:
        test_set = torchvision.datasets.SVHN(
            root=path, split="train", download=True, transform=transform_test
        )
        train_set.data = train_set.data[:-val_size]
        train_set.labels = train_set.labels[:-val_size]

        test_set.data = test_set.data[-val_size:]
        test_set.labels = test_set.labels[-val_size:]

    else:
        print("You are going to run models on the test set. Are you sure?")
        test_set = torchvision.datasets.SVHN(
            root=path, split="test", download=True, transform=transform_test
        )

    num_classes = 10

    return (
        {
            "train": torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "test": torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        },
        num_classes,
    )


class CIFAR10_Fouriered(torch.utils.data.Dataset):
    def __init__(self, base_ds_X, base_ds_Y, transform=None):
        self.data = base_ds_X
        self.base_ds_Y = base_ds_Y
        self.transform = transform

    def __getitem__(self, index):
        # do as you wish , add your logic here
        (img, label) = self.data[index], self.base_ds_Y[index]
        # for transformations for example
        if self.transform is not None:
            return self.transform(img), label
        return img, label

    def __len__(self):
        return len(self.data)


def loaders(
    dataset,
    path,
    batch_size,
    num_workers,
    transform_train,
    transform_test,
    use_validation=True,
    val_size=5000,
    use_data_size = None,
    split_classes=None,
    shuffle_train=True,
    fourier_features_params=None,
    **kwargs
):

    path = os.path.join(path, dataset.lower())
    dl_dict = dict()
    
    if dataset == "SVHN":
        return svhn_loaders(
            path,
            batch_size,
            num_workers,
            transform_train,
            transform_test,
            use_validation,
            val_size,
        )
    elif dataset == 'TinyImageNet100':
        print()
        print('USING TINYIMAGENET')
        return loaders_tinyimagenet(
            path,
            batch_size,
            num_workers,
            transform_train,
            transform_test,
            shuffle_train,
            limit_classes=100
        )
    elif dataset == 'TinyImageNet5':
        print()
        print('USING TINYIMAGENET (5 classes')
        return loaders_tinyimagenet(
            path,
            batch_size,
            num_workers,
            transform_train,
            transform_test,
            shuffle_train,
            limit_classes=5
        )
    elif dataset == 'MNIST5' or dataset == 'MNIST10':
        transform_test_mnist5 = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        
        train_dataset = datasets.MNIST(root='~/datasets', train=True, download=True, transform=transform_test_mnist5)
        test_dataset = datasets.MNIST(root='~/datasets', train=False, download=True, transform=transform_test_mnist5)

        # Filter the datasets to only include classes 0-4
        if dataset == 'MNIST5':
            classes = [0, 1, 2, 3, 4]
        elif dataset == 'MNIST10':
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        def filter_indices(dataset, classes):
            indices = []
            for i in range(len(dataset)):
                if dataset.targets[i] in classes:
                    indices.append(i)
            return indices

        train_indices = filter_indices(train_dataset, classes)
        test_indices = filter_indices(test_dataset, classes)

        # Create subsets of the datasets containing only the classes 0-4
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)

#         # Create data loaders for the train and test sets
#         train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
#         test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
        
        dl_dict = {
            "train": torch.utils.data.DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "test": torch.utils.data.DataLoader(
                test_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        }
        return dl_dict, 5
        
        
    elif dataset == 'Birds525':
        print()
        print('USING BIRDS 525')
        return loaders_birds525(
            path,
            batch_size,
            num_workers,
            transform_train,
            transform_test,
            shuffle_train
        )
    elif dataset == 'Birds525_200cl':
        print()
        print('USING BIRDS 525')
        return loaders_birds525(
            path,
            batch_size,
            num_workers,
            transform_train,
            transform_test,
            shuffle_train,
            limit_classes=200
        )
    elif dataset == 'CIFAR100Fake':
        ds = getattr(torchvision.datasets, 'CIFAR100')
    else:
        ds = getattr(torchvision.datasets, dataset)

    if dataset == "STL10":
        train_set = ds(
            root=path, split="train", download=True, transform=transform_train
        )
        num_classes = 10
        cls_mapping = np.array([0, 2, 1, 3, 4, 5, 7, 6, 8, 9])
        train_set.labels = cls_mapping[train_set.labels]
    elif dataset == "FakeData":
        train_set = FakeData(
            size=50000, image_size=(3, 32, 32), num_classes=100,
            transform=transform_train
        )
        num_classes=100
    else:
        train_set = ds(root=path, train=True, download=True, transform=transform_train)
        num_classes = max(train_set.targets) + 1

    if dataset == "CIFAR100Fake":
        print('Shuffling')
        from random import shuffle
        shuffle(train_set.targets)
        
    if use_data_size is not None:
        train_set.data = train_set.data[:use_data_size]
        train_set.targets = train_set.targets[:use_data_size]

    if use_validation:
        
        print('+'*60)
        print('+'*60)
        print()
        print('U S E   V A L I D A T I O N   =   T R U E')
        print()
        print('Are you sure you want 45 k + 5 k CIFAR split??')
        print()
        print('+'*60)
        print('+'*60)
        
        
        print(
            "Using train ("
            + str(len(train_set.data) - val_size)
            + ") + validation ("
            + str(val_size)
            + ")"
        )
        train_set.data = train_set.data[:-val_size]
        train_set.targets = train_set.targets[:-val_size]

        test_set = ds(root=path, train=True, download=True, transform=transform_test)
        test_set.train = False
        test_set.data = test_set.data[-val_size:]
        test_set.targets = test_set.targets[-val_size:]
        # delattr(test_set, 'data')
        # delattr(test_set, 'targets')
    else:
        print("You are going to run models on the test set. Are you sure?")
        if dataset == "STL10":
            test_set = ds(
                root=path, split="test", download=True, transform=transform_test
            )
            test_set.labels = cls_mapping[test_set.labels]
        elif dataset == "FakeData":
            test_set = FakeData(
                size=10000, image_size=(3, 32, 32), num_classes=100,
                transform=transform_train
            )
        else:
            test_set = ds(
                root=path, train=False, download=True, transform=transform_test
            )

    corrupt_train = kwargs.get("corrupt_train", None)
    if corrupt_train is not None and corrupt_train > 0:
        print("Train data corruption fraction:", corrupt_train)
        labels = np.array(train_set.targets)
        rs = np.random.RandomState(seed=228)
        mask = rs.rand(len(labels)) <= corrupt_train
        rnd_labels = rs.choice(num_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]
        assert len(train_set.targets) == len(labels)
        assert type(train_set.targets[0]) == type(labels[0])
        train_set.targets = labels
        
        return_train_subsets = kwargs.get("return_train_subsets", False)
        if return_train_subsets:
            corrupt_ids = np.arange(len(mask))[mask]
            normal_ids = np.arange(len(mask))[~mask]
            train_set_corrupt = torch.utils.data.Subset(train_set, corrupt_ids)
            train_set_normal = torch.utils.data.Subset(train_set, normal_ids)
            
            dl_dict.update({
                "train_corrupt": torch.utils.data.DataLoader(
                    train_set_corrupt,
                    batch_size=batch_size,
                    shuffle=shuffle_train,
                    num_workers=num_workers,
                    pin_memory=True,
                ),
                "train_normal": torch.utils.data.DataLoader(
                    train_set_normal,
                    batch_size=batch_size,
                    shuffle=shuffle_train,
                    num_workers=num_workers,
                    pin_memory=True,
                ),
            })

    if split_classes is not None:
        assert dataset == "CIFAR10"
        assert split_classes in {0, 1}

        print("Using classes:", end="")
        print(c10_classes[split_classes])
        train_mask = np.isin(train_set.targets, c10_classes[split_classes])
        train_set.data = train_set.data[train_mask, :]
        train_set.targets = np.array(train_set.targets)[train_mask]
        train_set.targets = np.where(
            train_set.targets[:, None] == c10_classes[split_classes][None, :]
        )[1].tolist()
        print("Train: %d/%d" % (train_set.data.shape[0], train_mask.size))

        test_mask = np.isin(test_set.targets, c10_classes[split_classes])
        print(test_set.data.shape, test_mask.shape)
        test_set.data = test_set.data[test_mask, :]
        test_set.targets = np.array(test_set.targets)[test_mask]
        test_set.targets = np.where(
            test_set.targets[:, None] == c10_classes[split_classes][None, :]
        )[1].tolist()
        print("Test: %d/%d" % (test_set.data.shape[0], test_mask.size))
    
    print("*****", train_set.data.shape)
    
    if fourier_features_params is not None:
        print('-'*80)
        print('Using Fourier Features: {}'.format(fourier_features_params))
        print('-'*80)
        
        slicer = FourierSlicer(size=fourier_features_params['size'], 
                               blocks=[fourier_features_params['blocks']], 
                               mask_mode=fourier_features_params['mask_mode'])
        print(type(train_set.data), train_set.data.dtype, train_set.data.min(), train_set.data.max())
        
        # ACTS LIKE transforms.TOTENSOR already
        new_train_data_X = next(slicer(torch.tensor(train_set.data).permute(0, 3, 1, 2) / 255.0 ))#.permute(0, 2, 3, 1).numpy()
        new_test_data_X = next(slicer(torch.tensor(test_set.data).permute(0, 3, 1, 2) / 255.0 ))
        
        new_train_set = CIFAR10_Fouriered(new_train_data_X, train_set.targets, transform_train)
        new_test_set = CIFAR10_Fouriered(new_test_data_X, test_set.targets, transform_test)
 
        
        train_set = new_train_set
        test_set = new_test_set
        
        print(type(train_set.data), train_set.data.dtype, train_set.data.min(), train_set.data.max())
#         train_set.data = next(slicer(torch.tensor(train_set.data).permute(0, 3, 1, 2))) / 255.0
#         test_set.data = next(slicer(torch.tensor(test_set.data).permute(0, 3, 1, 2))) / 255.0
    
    print('FINAL:')
    print(
        "Using train ("
        + str(len(train_set.data))
        + ") + validation ("
        + str(len(test_set.data))
        + ")"
    )
    
    dl_dict.update({
        "train": torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    })

    return dl_dict, num_classes



dir_structure_help = r"""
TinyImageNetPath
├── test
│   └── images
│       ├── test_0.JPEG
│       ├── t...
│       └── ...
├── train
│   ├── n01443537
│   │   ├── images
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   ├── n01629819
│   │   ├── images
│   │   │   ├── n01629819_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01629819_boxes.txt
│   ├── n...
│   │   ├── images
│   │   │   ├── ...
│   │   │   └── ...
├── val
│   ├── images
│   │   ├── val_0.JPEG
│   │   ├── v...
│   │   └── ...
│   └── val_annotations.txt
├── wnids.txt
└── words.txt
"""


def download_and_unzip(URL, root_dir):
    error_message = "Download is not yet implemented. Please, go to {URL} urself."
    raise NotImplementedError(error_message.format(URL))


def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while (img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


