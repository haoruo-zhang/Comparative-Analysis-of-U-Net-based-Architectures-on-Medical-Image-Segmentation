import os
import json
import torch
import nibabel as nib
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset


def get_data(data_pth, n_classes, crop_percent=0.1):
    data_config = json.load(open(os.path.join(data_pth, "dataset.json"), "r"))
    all_image = []
    all_label = []
    for pair in data_config["training"]:
        image = nib.load(os.path.join(data_pth, pair["image"][2:]))
        label = nib.load(os.path.join(data_pth, pair["label"][2:]))
        image = image.get_fdata()
        label = label.get_fdata()
        
        # 标准化图像
        image = (image - np.mean(image)) / np.std(image)
        slice_num = image.shape[2]
        
        # 按照比例居中裁剪 z 轴
        crop_slices = int(slice_num * crop_percent)  # 计算需要移除的切片数量
        start_slice = crop_slices  # 开始位置
        end_slice = slice_num - crop_slices  # 结束位置
        
        # 选择居中的部分切片
        image = image[:, :, start_slice:end_slice]
        label = label[:, :, start_slice:end_slice]
        
        # 调整图像和标签大小
        image = resize(image, (224, 224, image.shape[2]), mode="constant", cval=0)
        label = resize(label, (224, 224, label.shape[2]), mode="constant", cval=0)
        
        # 遍历切片并仅保留有标签的切片
        for i in range(image.shape[2]):  # 更新后的 slice_num
            label_slice = label[:, :, i]
            if np.sum(label_slice) > 0:  # 检查切片是否含有标签
                all_image.append(image[:, :, i])
                all_label.append(label[:, :, i])

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
