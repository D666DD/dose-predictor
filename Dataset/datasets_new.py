import numpy as np
import random
import pathlib
import os
import torch
import gzip
import torch.utils.data as data
from PIL import Image

file_list = ["ct.csv", "dose.csv", "possible_dose_mask.csv", "PTV56.csv", "PTV63.csv", "PTV70.csv",
             "Brainstem.csv", "SpinalCord.csv", "RightParotid.csv", "LeftParotid.csv",
             "Esophagus.csv", "Larynx.csv", "Mandible.csv"]
organ_list = ["Brainstem.csv", "SpinalCord.csv", "RightParotid.csv", "LeftParotid.csv",
             "Esophagus.csv", "Larynx.csv", "Mandible.csv"]             


class MyDataset(data.Dataset):
    def __init__(self, phase, aug=False):
        self.phase = phase
        self.aug = aug
        # self.transform = {'train': train_transform, 'val': val_transform}
        self.list_case_id = {'train': r'Data/OpenKBP_NEW/train-pats',
                             'val': r'Data/OpenKBP_NEW/validation-pats',
                             'test': r'Data/OpenKBP_NEW/test-pats'}[phase]
        self.list_datapath = []

        for case_id in os.listdir(self.list_case_id):
            path = os.path.join(self.list_case_id, case_id)
            for i in range(128):
              self.list_datapath.append([path,i])
            """""
            self.list_datapath.append(case_id)
                # print(slice_num)
            for i in range(128):
                for file_name in organ_list:
                    file_name = file_name.split(".")[0]
                    
                    self.list_datapath.append()
                    #print(os.path.join(path, file_name + "_" + str(i) + ".npy.gz"))
                    #with gzip.GzipFile(os.path.join(path, file_name + "_" + str(i) + ".npy.gz"), 'rb') as f:
                        #data = np.load(f)
                        #print(file_name)
                    imagepath = os.path.join(path, "ct_" + str(slice_num[i]) + ".npy")  # 0
                    dosepath = os.path.join(path, "dose_" + str(slice_num[i]) + ".npy")  # 1
                    maskpath = os.path.join(path, "mask_" + str(slice_num[i]) + ".npy")  # 2
                    ptv_56 = os.path.join(path, "ptv56_" + str(slice_num[i]) + ".npy")  # 3
                    ptv_63 = os.path.join(path, "ptv63_" + str(slice_num[i]) + ".npy")  # 4
                    ptv_70 = os.path.join(path, "ptv70_" + str(slice_num[i]) + ".npy")  # 5
                    oars = os.path.join(path, "oar_" + str(slice_num[i]) + ".npy")  # 6
                    self.list_datapath.append([imagepath, dosepath, maskpath, ptv_56, ptv_63, ptv_70, oars])
                    """""
                    
        if phase == "train":
            random.shuffle(self.list_datapath)

        self.sum_case = len(self.list_datapath)

    def __getitem__(self, index_):
        # get one sample based on the index
        if index_ <= self.sum_case - 1:
            datapath = self.list_datapath[index_]
        else:
            new_index_ = index_ - (index_ // self.sum_case) * self.sum_case
            datapath = self.list_datapath[new_index_]

        # npimage = np.load(datapath[0])
        # npimage = np.reshape(npimage, (128,128,1)).transpose((2, 0, 1))
        
        path = datapath[0]
        slice_num = datapath[1]
        
        ct = np.zeros((128, 128, 1))
        dose = np.zeros((128, 128, 1))
        possible_mask = np.zeros((128, 128, 1))
        oar = np.zeros((128, 128, 1))
        ptv = np.zeros((128, 128, 1))
        
        # CT
        with gzip.GzipFile(os.path.join(path, "ct" + "_" + str(slice_num) + ".npy.gz"), 'rb') as f:
            data = np.load(f)
        ct = np.clip(data, 0, 2500) / 1250.0 - 1
        ct = np.reshape(ct, (128, 128, 1))
        
        # DOSE
        with gzip.GzipFile(os.path.join(path, "dose" + "_" + str(slice_num) + ".npy.gz"), 'rb') as f:
            data = np.load(f)
        dose = np.clip(data, 0, 80) / 40.0 - 1
        #dose = np.reshape(dose, (128, 128, 1))
        
        # POSSIBLE
        with gzip.GzipFile(os.path.join(path, "possible_dose_mask" + "_" + str(slice_num) + ".npy.gz"), 'rb') as f:
            data = np.load(f)
        possible_mask = np.reshape(data, (128, 128, 1))
        
        
        # OAR
        for oar_name in organ_list:
            oar_name = oar_name.split(".")[0]
            with gzip.GzipFile(os.path.join(path, oar_name + "_" + str(slice_num) + ".npy.gz"), 'rb') as f:
                data = np.load(f)
            oar += np.reshape(data, (128, 128, 1))
        
        
        # PTV
        with gzip.GzipFile(os.path.join(path, "PTV70" + "_" + str(slice_num) + ".npy.gz"), 'rb') as f:
            data = np.load(f)
        ptv += 70.0/70.0 *  np.reshape(data, (128, 128, 1))
        with gzip.GzipFile(os.path.join(path, "PTV63" + "_" + str(slice_num) + ".npy.gz"), 'rb') as f:
            data = np.load(f)
        ptv += 63.0/70.0 *  np.reshape(data, (128, 128, 1))
        with gzip.GzipFile(os.path.join(path, "PTV56" + "_" + str(slice_num) + ".npy.gz"), 'rb') as f:
            data = np.load(f)
        ptv += 56.0/70.0 *  np.reshape(data, (128, 128, 1)) 
        
        
        
        
        image = np.concatenate([ct, ptv, possible_mask, oar], axis=-1).transpose((2, 0, 1))

        image_ = torch.from_numpy(image.copy()).float()
        dose_ = torch.from_numpy(dose.copy()).float()
        mask_ = torch.from_numpy(possible_mask.copy()).float().squeeze()
        #print(image_.shape)
        return image_, dose_, mask_, datapath[1], datapath[0]

    def __len__(self):
        return self.sum_case

#train = MyDataset('train')
#train.__getitem__(76)