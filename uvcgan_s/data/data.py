import os
import torch

import torchvision

from torch.utils.data import DataLoader, DistributedSampler

from uvcgan_s.consts import (
    ROOT_DATA, SPLIT_TRAIN, MERGE_PAIRED, MERGE_UNPAIRED
)
from uvcgan_s.torch.select      import extract_name_kwargs
from uvcgan_s.torch.distributed import is_distributed

from .datasets.celeba                 import CelebaDataset
from .datasets.image_domain_folder    import ImageDomainFolder
from .datasets.image_domain_hierarchy import ImageDomainHierarchy
from .datasets.zipper                 import DatasetZipper
from .datasets.ndarray_domain_hierarchy import NDArrayDomainHierarchy
from .datasets.h5array_domain_hierarchy import H5ArrayDomainHierarchy
from .datasets.toy_mix_blur_dataset   import ToyMixBlurDataset

from .loader_zipper import DataLoaderZipper
from .transforms    import select_transform

def select_dataset(name, path, split, transform, **kwargs):
    # pylint: disable=too-many-return-statements
    # pylint: disable=too-many-branches

    if name == 'celeba':
        return CelebaDataset(
            path, transform = transform, split = split, **kwargs
        )

    if name in [ 'cyclegan', 'image-domain-folder' ]:
        return ImageDomainFolder(
            path, transform = transform, split = split, **kwargs
        )

    if name in [ 'image-domain-hierarchy' ]:
        return ImageDomainHierarchy(
            path, transform = transform, split = split, **kwargs
        )

    if name == 'imagenet':
        return torchvision.datasets.ImageNet(
            path, transform = transform, split = split, **kwargs
        )

    if name in [ 'imagedir', 'image-folder' ]:
        return torchvision.datasets.ImageFolder(
            os.path.join(path, split), transform = transform, **kwargs
        )

    if name == 'ndarray-domain-hierarchy':
        return NDArrayDomainHierarchy(
            path, transform = transform, split = split, **kwargs
        )

    if name == 'h5array-domain-hierarchy':
        return H5ArrayDomainHierarchy(
            path, transform = transform, split = split, **kwargs
        )

    if name == 'toy-mix-blur':
        return ToyMixBlurDataset(
            path, transform = transform, split = split, **kwargs
        )

    raise ValueError(f"Unknown dataset: '{name}'")

def construct_single_dataset(dataset_config, split):
    name, kwargs = extract_name_kwargs(dataset_config.dataset)
    path         = os.path.join(ROOT_DATA, kwargs.pop('path', name))

    if split == SPLIT_TRAIN:
        transform = select_transform(dataset_config.transform_train)
    else:
        transform = select_transform(dataset_config.transform_test)

    return select_dataset(name, path, split, transform, **kwargs)

def construct_datasets(data_config, split):
    return [
        construct_single_dataset(config, split)
            for config in data_config.datasets
    ]

def construct_single_loader(
    dataset, batch_size, shuffle,
    workers         = None,
    prefetch_factor = 2,
    drop_last       = False,
    **kwargs
):
    # pylint: disable=too-many-arguments
    if workers is None:
        workers = min(torch.get_num_threads(), 20)

    sampler = None
    if is_distributed():
        # each process draws its own disjoint shard of the dataset
        sampler = DistributedSampler(
            dataset, shuffle = shuffle, drop_last = drop_last
        )
        shuffle = False

    if workers > 0:
        # torch >= 2 rejects prefetch_factor without workers
        kwargs['prefetch_factor'] = prefetch_factor

    return DataLoader(
        dataset, batch_size,
        shuffle     = shuffle,
        sampler     = sampler,
        num_workers = workers,
        pin_memory  = True,
        drop_last   = drop_last,
        **kwargs
    )

def set_loader_epoch(loader, epoch):
    """Reseed distributed samplers so that shuffling differs per epoch."""
    if isinstance(loader, (list, tuple)):
        for x in loader:
            set_loader_epoch(x, epoch)
        return

    if hasattr(loader, 'set_epoch'):
        loader.set_epoch(epoch)
        return

    sampler = getattr(loader, 'sampler', None)
    if hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch)

def construct_data_loaders(data_config, batch_size, split):
    datasets = construct_datasets(data_config, split)
    shuffle  = (split == SPLIT_TRAIN)

    if data_config.merge_type == MERGE_PAIRED:
        dataset = DatasetZipper(datasets)

        return construct_single_loader(
            dataset, batch_size, shuffle, data_config.workers,
            drop_last = False
        )

    loaders = [
        construct_single_loader(
            dataset, batch_size, shuffle, data_config.workers,
            drop_last = (data_config.merge_type == MERGE_UNPAIRED)
        ) for dataset in datasets
    ]

    if data_config.merge_type == MERGE_UNPAIRED:
        return DataLoaderZipper(loaders)

    if len(loaders) == 1:
        return loaders[0]

    return loaders

