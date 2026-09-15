import os

import numpy as np
from torch.utils.data import Dataset

from uvcgan_s.consts import SPLIT_TRAIN

DSET_INDEX = 'index'
DSET_DATA  = 'data'

H5_EXT = [ '', '.h5', '.hdf5' ]


class H5ArrayDomainHierarchy(Dataset):

    def __init__(
        self, path, domain,
        split     = SPLIT_TRAIN,
        transform = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self._path = None
        path_base  = os.path.join(path, split, domain)

        for ext in H5_EXT:
            path = path_base + ext
            if os.path.exists(path):
                self._path = path
                break
        else:
            raise RuntimeError(f"Failed to find h5 dataset '{path_base}'")

        # pylint: disable=import-outside-toplevel
        import h5py

        with h5py.File(self._path, 'r') as f:
            self._len = len(f[DSET_DATA])

        # NOTE: h5py handles are not fork-safe, so the file is opened
        #       lazily by whichever process (e.g. DataLoader worker) reads
        #       it first. This also keeps the dataset picklable.
        self._f    = None
        self._dset = None

        self._transform = transform

    def _ensure_open(self):
        if self._dset is None:
            # pylint: disable=import-outside-toplevel
            import h5py

            self._f    = h5py.File(self._path, 'r')
            self._dset = self._f[DSET_DATA]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        self._ensure_open()
        result = np.float32(self._dset[index])

        if self._transform is not None:
            result = self._transform(result)

        return result

