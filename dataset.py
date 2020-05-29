# -------------------------------------------------------------------------------
# Written by: Rilwan Remilekun Basaru
# -------------------------------------------------------------------------------

from __future__ import print_function
import numpy as np


class DataLoader:
    def __init__(self, ite_index, shuffle=True):
        """

        :param ite_index: list of dataset indices can be used in the 'next_img()' function to iterate through images
         for training or testing
        :param shuffle: boolean to indicate whether to shuffle the dataset
        """
        self.ite = None
        self.reset_ite()

        self.src_img_data = None

        if ite_index is not None and len(ite_index) > 0:
            if shuffle:
                np.random.shuffle(ite_index)
            self.ite_index = ite_index

        self.size = len(self.ite_index)

    def ite_agu(self, ite):
        return self.ite_index[ite]

    def next_img(self, batch_size):
        """
        Get the next image in the image list in NHWC format
        :return: 4D-RGB image, 4D-Ground truth image label and 4D-weight(nullable)
        """
        # TODO: Implement function for loading data pair based on batch size
        raise NotImplementedError

    def is_next(self):
        return self.ite < self.size

    def reset_ite(self):
        self.ite = 0

    def set_ite(self, iter):
        if iter > self.size:
            iter = self.size
        self.ite = iter
