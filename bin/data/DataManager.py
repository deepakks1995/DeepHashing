from keras.preprocessing import image
import random, os
import numpy as np


class DataManager(object):
    """Class to handle data, generate batches of different sizes"""

    def __init__(self, path, batch_size=125, steps_per_epoch=2, tuple=False, similar_samples=8, model_vars=None):
        super(DataManager, self).__init__()
        self.path = path
        self.current_idx = 0
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.curr_steps_per_epoch = 0  # type: int
        self.similar_samples = similar_samples
        self.folder_counter = []
        self.total_subjects = 0
        self.model_vars = model_vars
        self.__init__vars()

    def __init__vars(self):
        if self.batch_size > 1:
            raise ValueError("Current implementation cannot handle batch size greater than 1")

        class_folders = os.listdir(os.getcwd() + "/" + self.path)
        for folder in class_folders:
            listings = os.listdir(os.getcwd() + "/" + self.path + "/" + folder)
            self.total_subjects += int(len(listings) / self.similar_samples)
            self.folder_counter.append((folder, int(len(listings) / self.similar_samples)))

    def has_next(self):
        return self.curr_steps_per_epoch < self.steps_per_epoch

    def on_epoch_end(self):
        self.current_idx = 0
        self.curr_steps_per_epoch = 0

    def __get_folder(self, current_idx):
        temp = 0
        idx = 0
        for (Key, Value) in self.folder_counter:
            idx = current_idx - temp
            temp += Value
            if temp >= current_idx:
                if idx == 0:
                    idx = Value
                return Key, idx
        raise ValueError("Current Index out of Bounds")

    def next_batch(self):
        self.curr_steps_per_epoch += 1
        self.current_idx = (self.current_idx + 1) % self.total_subjects

        prim_batch = [[np.zeros((self.batch_size, 224, 224, 3))] +
                      [np.zeros((self.batch_size, self.model_vars.kbit)) for _ in range(3)] +
                      [np.zeros(self.batch_size)]] + \
                     [[np.zeros((self.batch_size, 224, 224, 3))] +
                      [np.zeros((self.batch_size, self.model_vars.kbit)) for _ in range(3)] +
                      [np.zeros(self.batch_size)]]

        sec_batch = [[np.zeros((self.batch_size, 224, 224, 3))] +
                    [np.zeros((self.batch_size, self.model_vars.kbit)) for _ in range(3)] +
                    [np.zeros(self.batch_size)]] + \
                    [[np.zeros((self.batch_size, 224, 224, 3))] +
                    [np.zeros((self.batch_size, self.model_vars.kbit)) for _ in range(3)] +
                    [np.zeros(self.batch_size)]]

        index_list = []

        for itr in range(self.batch_size):
            if itr % 2 == 1:
                continue
            anchor_image, anchor_idx = self.__get_anchor_image(self.current_idx)
            positive_image, pos_idx = self.__get_pos_images(self.current_idx)
            negative_image, neg_idx = self.__get_neg_images(self.current_idx)

            prim_batch[0][0][itr, :, :, :] = anchor_image
            prim_batch[0][1][itr, :] = self.model_vars.B[:, anchor_idx]
            prim_batch[0][2][itr, :] = self.model_vars.B[:, pos_idx]
            prim_batch[0][3][itr, :] = self.model_vars.V[:, pos_idx]
            prim_batch[0][4][itr] = 1

            prim_batch[1][0][itr, :, :, :] = anchor_image
            prim_batch[1][1][itr, :] = self.model_vars.B[:, anchor_idx]
            prim_batch[1][2][itr, :] = self.model_vars.B[:, neg_idx]
            prim_batch[1][3][itr, :] = self.model_vars.V[:, neg_idx]
            prim_batch[1][4][itr] = -1

            sec_batch[0][0][itr, :, :, :] = positive_image
            sec_batch[0][1][itr, :] = self.model_vars.B[:, pos_idx]
            sec_batch[0][2][itr, :] = self.model_vars.B[:, anchor_idx]
            sec_batch[0][3][itr, :] = self.model_vars.U[:, anchor_idx]
            sec_batch[0][4][itr] = 1

            sec_batch[0][0][itr, :, :, :] = negative_image
            sec_batch[0][1][itr, :] = self.model_vars.B[:, neg_idx]
            sec_batch[0][2][itr, :] = self.model_vars.B[:, anchor_idx]
            sec_batch[0][3][itr, :] = self.model_vars.U[:, anchor_idx]
            sec_batch[0][4][itr] = -1

            index_list.append([anchor_idx, pos_idx])
            index_list.append([anchor_idx, neg_idx])

        return prim_batch, sec_batch, index_list

    def __get_anchor_image(self, current_idx):
        folder_path, idx = self.__get_folder(current_idx)
        i = 1
        anchor_image = image.load_img(
            os.getcwd() + "/" + self.path + "/" + folder_path + "/" + (str(idx) + "_" + str(i)) + '.png')
        anchor_image = image.img_to_array(anchor_image)
        anchor_image = np.expand_dims(anchor_image, axis=0)
        anchor_idx = (current_idx - 1) * self.similar_samples + (i - 1)
        return anchor_image, anchor_idx

    def __get_pos_images(self, current_idx):

        folder_path, idx = self.__get_folder(current_idx)
        i = random.randint(1, self.similar_samples)
        positive_image = image.load_img(
            os.getcwd() + "/" + self.path + "/" + folder_path + "/" + (str(idx) + "_" + str(i)) + '.png')
        positive_image = image.img_to_array(positive_image)
        positive_image = np.expand_dims(positive_image, axis=0)
        pos_index = (current_idx - 1) * self.similar_samples + (i - 1)

        return positive_image, pos_index

    def __get_neg_images(self, current_idx):

        random_list = random.sample(range(1, self.total_subjects), 2)
        negative_image = None
        neg_idx = None
        for i in random_list:
            if i == current_idx:
                continue
            folder_path, idx = self.__get_folder(i)

            negative_image = image.load_img(
                os.getcwd() + "/" + self.path + "/" + str(folder_path) + "/" + (str(idx) + "_1") + '.png')
            negative_image = image.img_to_array(negative_image)
            negative_image = np.expand_dims(negative_image, axis=0)
            neg_idx = (i - 1) * self.similar_samples

        return negative_image, neg_idx
