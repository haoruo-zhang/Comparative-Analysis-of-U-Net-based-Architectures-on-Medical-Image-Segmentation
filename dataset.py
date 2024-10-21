import os
import json
import torch
import nibabel as nib
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset


def get_data(data_pth, n_classes):
    data_config = json.load(open(os.path.join(data_pth, "dataset.json"), "r"))
    all_image = []
    all_label = []
    for pair in data_config["training"]:
        image = nib.load(os.path.join(data_pth, pair["image"][2:]))
        label = nib.load(os.path.join(data_pth, pair["label"][2:]))
        image = image.get_fdata()
        label = label.get_fdata()
        image = (image - np.mean(image)) / np.std(image)
        slice_num = image.shape[2]
        image = resize(image, (224, 224, slice_num), mode="constant", cval=0)
        label = resize(label, (224, 224, slice_num), mode="constant", cval=0)
        # label_new = np.zeros((n_classes, 224, 224, slice_num))
        # for i in range(n_classes):
        #     label_new[i, :, :, :] = label[:, :, :] == i
        for i in range(slice_num):
            all_image.append(image[:, :, i])
            all_label.append(label[:, :, i])
            # all_label.append(label_new[:, :, :, i])
    all_image = np.array(all_image)
    all_label = np.array(all_label)
    return all_image, all_label


class ImageDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = torch.tensor(self.data[index], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.label[index], dtype=torch.float32).unsqueeze(0)
        if self.transform:
            cat_image = torch.cat((image, label), 0)
            cat_image = self.transform(cat_image)
            image = cat_image[0, :, :].unsqueeze(0)
            label = cat_image[1, :, :].unsqueeze(0)
        label = label.squeeze(0)
        sample = {"data": image, "label": label, "index": index}
        return sample
