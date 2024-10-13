import torch
import torchvision
import os
import numpy as np

class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        train=True,
        index_targets=False
    ):
        """
        ImageNet

        Dataset wrapper for the ImageNet Tiny dataset.

        :param root: root directory for ImageNet Tiny data
        :param transform: transformations to apply to images
        :param train: whether to load train data (or validation)
        :param index_targets: whether to index the id of each labeled image
        """

        suffix = 'train/' if not train else 'val/'  # Adjust this for test if needed
        data_path = os.path.join(root, suffix)  # Adjusted path
        print(f'data-path {data_path}')

        super(ImageNet, self).__init__(root=data_path, transform=transform)
        print('Initialized ImageNet')

        if index_targets:
            self.targets = []
            for sample in self.samples:
                self.targets.append(sample[1])
            self.targets = np.array(self.targets)
            self.samples = np.array(self.samples)

            mint = None
            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(
                    self.targets == t)).tolist()
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(mint, len(indices))

def make_imagenet1k(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    training=True,
    drop_last=True,
):
    dataset = ImageNet(
        root=root_path,
        transform=transform,
        train=training,
        index_targets=False)
    print('ImageNet dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    print('[INFO] ImageNet unsupervised data loader created')

    return dataset, data_loader, dist_sampler