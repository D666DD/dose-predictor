import numpy as np
import random
import pathlib
import os
import torch
import torch.utils.data as data
from PIL import Image
class MyDataset(data.Dataset):
    def __init__(self, phase,aug=False):
        self.phase = phase
        self.aug = aug
        # self.transform = {'train': train_transform, 'val': val_transform}
        self.list_case_id = {'train': r'Data/OpenKBP3/train-pats',
                             'val': r'Data/OpenKBP3/validation-pats',
                             'test': r'Data/OpenKBP3/test-pats'}[phase]
        self.list_datapath = []
        
        #if phase == "test":
        #  file_list_sorted = sorted(os.listdir(self.list_case_i))
        for case_id in os.listdir(self.list_case_id):
            path=os.path.join(self.list_case_id,case_id)
            list_fn = pathlib.Path(path).glob("slice_record.npy")

            for fn in list_fn:
                """""
                dir_path, file_name = os.path.split(fn)
                base_name = os.path.splitext(file_name)[0]
                base_name = base_name.replace('_structure_image', '')
                n_slice = os.path.join(dir_path, base_name)
                """""
                slice_num = np.load(fn)
                #print(slice_num)
                for i in range(slice_num.shape[0]):
                    imagepath = os.path.join(path, "ct_" + str(slice_num[i]) + ".npy") #0
                    dosepath = os.path.join(path, "dose_" + str(slice_num[i]) + ".npy") #1
                    maskpath = os.path.join(path, "mask_" + str(slice_num[i]) + ".npy") #2
                    dm_56 = os.path.join(path, "dm56_" + str(slice_num[i]) + ".npy") #3
                    dm_63 = os.path.join(path, "dm63_" + str(slice_num[i]) + ".npy") #4
                    dm_70 = os.path.join(path, "dm70_" + str(slice_num[i]) + ".npy") #5
                    dm_oars = os.path.join(path, "dmoar_" + str(slice_num[i]) + ".npy") #6
                    self.list_datapath.append([imagepath, dosepath, maskpath, dm_56, dm_63, dm_70,dm_oars])
                #print(self.list_datapath)

        if phase == "train":
          random.shuffle(self.list_datapath)
        self.sum_case = len(self.list_datapath)
        print(self.sum_case)

    def __getitem__(self, index_):
        # 按索引获取数据集中的一个样本
        if index_ <= self.sum_case - 1:
            datapath = self.list_datapath[index_]
        else:
            new_index_ = index_ - (index_ // self.sum_case) * self.sum_case
            datapath = self.list_datapath[new_index_]

        #npimage = np.load(datapath[0])
        #npimage = np.reshape(npimage, (128,128,1)).transpose((2, 0, 1))
        npimage = np.concatenate([
                  np.reshape(np.load(datapath[0]), (128, 128, 1)),
                  np.reshape(np.load(datapath[6]), (128, 128, 1)),
                  np.reshape(np.load(datapath[3]), (128, 128, 1)),
                  np.reshape(np.load(datapath[4]), (128, 128, 1)),
                  np.reshape(np.load(datapath[5]), (128, 128, 1))], axis=-1).transpose((2, 0, 1))
        
        npdose = np.load(datapath[1])
        npmask = np.load(datapath[2])
        
        image_ = torch.from_numpy(npimage.copy()).float()
        dose_ = torch.from_numpy(npdose.copy()).float()
        mask_=torch.from_numpy(npmask.copy()).float()
        
        return image_,  dose_,mask_,datapath[0],datapath[1]

    def __len__(self):
        return self.sum_case

#train = MyDataset('train')
#train.__getitem__(76)