from torch.utils.data import Dataset
import os
import torch
import numpy as np
import cv2
from albumentations import (ShiftScaleRotate, Compose, HorizontalFlip, VerticalFlip)

def gaussian_function(distance, sigma=1.0):
    return np.exp(-0.5 * (distance / sigma) ** 2)

class MyDataset(Dataset):
    def __init__(self, phase, aug=False, data_root='data'):
        self.phase = phase
        self.aug = aug
        # self.transform = {'train': train_transform, 'val': val_transform}
        self.list_case_id = {'train': r'Data/OpenKBP_NPY/train',
                             'val': r'Data/OpenKBP_NPY/validation',
                             'test': r'Data/OpenKBP_NPY/test'}[phase]
        self.file_dir_list = []
        self.file_name_list = []
        self.file_dir_list = self.file_dir_list + [self.list_case_id] * len(
            os.listdir(os.path.join(self.list_case_id, 'ct')))
        #self.file_name_list.extend(os.listdir(os.path.join(self.list_case_id, 'ct')))
        if self.phase == 'test':
            for i in range(100):
                for j in range(128):
                    id = 241 + i
                    file_name = "pt_" + str(id) + "_" + str(j) + ".npy"
                    self.file_name_list.append(file_name)
        else:
            self.file_name_list.extend(os.listdir(os.path.join(self.list_case_id, 'ct')))
        
        self.len = len(self.file_name_list)

    def __getitem__(self, idx):
        file_dir = self.file_dir_list[idx]
        file_name = self.file_name_list[idx]
        ct = np.load(os.path.join(file_dir, 'ct', file_name))[:, :, np.newaxis]
        dose = np.load(os.path.join(file_dir, 'dose', file_name))[:, :, np.newaxis]

        Mask_Brainstem = np.load(os.path.join(file_dir, 'Mask_Brainstem', file_name))[:, :, np.newaxis]
        Mask_Esophagus = np.load(os.path.join(file_dir, 'Mask_Esophagus', file_name))[:, :, np.newaxis]
        Mask_Larynx = np.load(os.path.join(file_dir, 'Mask_Larynx', file_name))[:, :, np.newaxis]
        Mask_LeftParotid = np.load(os.path.join(file_dir, 'Mask_LeftParotid', file_name))[:, :, np.newaxis]
        Mask_Mandible = np.load(os.path.join(file_dir, 'Mask_Mandible', file_name))[:, :, np.newaxis]
        Mask_possible_dose_mask = np.load(os.path.join(file_dir, 'Mask_possible_dose_mask', file_name))[:, :,
                                  np.newaxis]
        Mask_PTV56 = np.load(os.path.join(file_dir, 'Mask_PTV56', file_name))[:, :, np.newaxis]
        Mask_PTV63 = np.load(os.path.join(file_dir, 'Mask_PTV63', file_name))[:, :, np.newaxis]
        Mask_PTV70 = np.load(os.path.join(file_dir, 'Mask_PTV70', file_name))[:, :, np.newaxis]
        Mask_RightParotid = np.load(os.path.join(file_dir, 'Mask_RightParotid', file_name))[:, :, np.newaxis]
        Mask_SpinalCord = np.load(os.path.join(file_dir, 'Mask_SpinalCord', file_name))[:, :, np.newaxis]
        
        PSDM_Brainstem = np.load(os.path.join(file_dir, 'PSDM_Brainstem', file_name))[:, :, np.newaxis]
        PSDM_Esophagus = np.load(os.path.join(file_dir, 'PSDM_Esophagus', file_name))[:, :, np.newaxis]
        PSDM_Larynx = np.load(os.path.join(file_dir, 'PSDM_Larynx', file_name))[:, :, np.newaxis]
        PSDM_LeftParotid = np.load(os.path.join(file_dir, 'PSDM_LeftParotid', file_name))[:, :, np.newaxis]
        PSDM_Mandible = np.load(os.path.join(file_dir, 'PSDM_Mandible', file_name))[:, :, np.newaxis]
        PSDM_possible_dose_mask = np.load(os.path.join(file_dir, 'PSDM_possible_dose_mask', file_name))[:, :, np.newaxis]
        PSDM_PTV56 = np.load(os.path.join(file_dir, 'PSDM_PTV56', file_name))[:, :, np.newaxis]
        PSDM_PTV63 = np.load(os.path.join(file_dir, 'PSDM_PTV63', file_name))[:, :, np.newaxis]
        PSDM_PTV70 = np.load(os.path.join(file_dir, 'PSDM_PTV70', file_name))[:, :, np.newaxis]
        PSDM_RightParotid = np.load(os.path.join(file_dir, 'PSDM_RightParotid', file_name))[:, :, np.newaxis]
        PSDM_SpinalCord = np.load(os.path.join(file_dir, 'PSDM_SpinalCord', file_name))[:, :, np.newaxis]
        
        GSDM_Brainstem = np.load(os.path.join(file_dir, 'GSDM_Brainstem', file_name))[:, :, np.newaxis]
        GSDM_Esophagus = np.load(os.path.join(file_dir, 'GSDM_Esophagus', file_name))[:, :, np.newaxis]
        GSDM_Larynx = np.load(os.path.join(file_dir, 'GSDM_Larynx', file_name))[:, :, np.newaxis]
        GSDM_LeftParotid = np.load(os.path.join(file_dir, 'GSDM_LeftParotid', file_name))[:, :, np.newaxis]
        GSDM_Mandible = np.load(os.path.join(file_dir, 'GSDM_Mandible', file_name))[:, :, np.newaxis]
        GSDM_possible_dose_mask = np.load(os.path.join(file_dir, 'GSDM_possible_dose_mask', file_name))[:, :, np.newaxis]
        GSDM_PTV56 = np.load(os.path.join(file_dir, 'GSDM_PTV56', file_name))[:, :, np.newaxis]
        GSDM_PTV63 = np.load(os.path.join(file_dir, 'GSDM_PTV63', file_name))[:, :, np.newaxis]
        GSDM_PTV70 = np.load(os.path.join(file_dir, 'GSDM_PTV70', file_name))[:, :, np.newaxis]
        GSDM_RightParotid = np.load(os.path.join(file_dir, 'GSDM_RightParotid', file_name))[:, :, np.newaxis]
        GSDM_SpinalCord = np.load(os.path.join(file_dir, 'GSDM_SpinalCord', file_name))[:, :, np.newaxis]
        

        PTVs_mask = 70.0 / 70. * Mask_PTV70 + 63.0 / 70. * Mask_PTV63 + 56.0 / 70. * Mask_PTV56
        dose = (dose + 1.0) * 40.0 / 80.0
        ct = (ct + 1.0) * 1250.0 / 2500.0
        
        #print(PSDM_possible_dose_mask.max(), PSDM_possible_dose_mask.min(), GSDM_possible_dose_mask.max(), GSDM_possible_dose_mask.min())
        
        
        data_all = np.concatenate(
            [dose, ct, PTVs_mask, Mask_Brainstem, Mask_Esophagus, Mask_Larynx, Mask_LeftParotid, Mask_Mandible,
             Mask_possible_dose_mask, Mask_RightParotid, Mask_SpinalCord, 
                         PSDM_Brainstem, PSDM_Esophagus, PSDM_Larynx,
                         PSDM_LeftParotid, PSDM_Mandible, PSDM_possible_dose_mask, PSDM_PTV56, PSDM_PTV63, PSDM_PTV70,
                         PSDM_RightParotid, PSDM_SpinalCord,
                         GSDM_Brainstem, GSDM_Esophagus, GSDM_Larynx,
                         GSDM_LeftParotid, GSDM_Mandible, GSDM_possible_dose_mask, GSDM_PTV56, GSDM_PTV63, GSDM_PTV70,
                         GSDM_RightParotid, GSDM_SpinalCord], axis=-1)
             
        dose = torch.from_numpy(data_all[:, :, 0:1]).permute(2, 0, 1).float()
        ct = torch.from_numpy(data_all[:, :, 1:2]).permute(2, 0, 1).float()
        dis = torch.from_numpy(data_all[:, :, 1:]).permute(2, 0, 1).float()
        
        #print(dose.min(), dose.max(), ct.min(), ct.max())
        return dis, dose, ct, file_dir, file_name
        # data, target,mask,cbct_path,ct_path

    def __len__(self):
        return self.len
        
tr_train = MyDataset("train")

tr_train.__getitem__(64)