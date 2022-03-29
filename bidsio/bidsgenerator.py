from bidsloader import BIDSLoader
import numpy as np


class BIDSGenerator:
    def __init__(self,
                 loader: BIDSLoader = None):
        '''
        Creates a generator from a BIDSLoader object.
        Parameters
        ----------
        loader
        '''
        self.loader = loader
        return

    def generate_batches(self,
                         idx_range: tuple = None,
                         data_only: bool = False):
        '''
        Generator that yields one batch per iteration, determined by the loader.batch_size. Returns samples within the
        specified range of indies.
        Parameters
        ----------
        idx_range : tuple (int)
            Start and end range of data to load
        data_only : bool
            If true, load only from data_list and don't return samples from target_list.

        Yields
        ------
        np.array
            Array of shape (batch_size, num_data, *image.shape) containing data.
        np.array
            Array of shape (batch_size, num_target, *image.shape) containing targets.
        '''
        if(idx_range is None):
            start_idx = 0
            end_idx = len(self.loader)
        else:
            start_idx, end_idx = idx_range

        batch_size = self.loader.batch_size
        for i in range(start_idx, end_idx, batch_size):
            max_batch_idx = np.min([i+batch_size])
            yield self.loader.load_batch(range(i, max_batch_idx), data_only=data_only)
        return

    def generate_batch_with_image(self,
                                  idx_range: tuple = None):
        '''
        Generator that yields the image data along with the BIDS image file. Useful for writing images with similar
        BIDS entities.
        Yields
        ------
        np.array
            Array of shape (batch_size, num_data, *image.shape) containing the data from loader.data_list
        list [BIDSImageFile)
            BIDSImageFile corresponding to the data loaded in the array.
        '''
        if(idx_range is None):
            start_idx = 0
            end_idx = len(self.loader)
        else:
            start_idx, end_idx = idx_range

        batch_size = self.loader.batch_size
        for i in range(start_idx, end_idx, batch_size):
            max_batch_idx = np.min([i+self.batch_size, len(self.loader)])
            image_list = [self.loader.data_list[j] for j in range(i, max_batch_idx)]
            yield self.loader.load_batch(range(i, max_batch_idx), data_only=True), image_list
        return
