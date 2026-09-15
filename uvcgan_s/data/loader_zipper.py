
class DataLoaderZipper:

    def __init__(self, loaders):
        self._loaders = loaders

    def __len__(self):
        return min(len(d) for d in self._loaders)

    def __iter__(self):
        return zip(*self._loaders)

    def set_epoch(self, epoch):
        for loader in self._loaders:
            sampler = getattr(loader, 'sampler', None)
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)

