import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os, math
import sys

class SavedDataset(Dataset):
    def __init__(self, path_dir):
        self.data_path = path_dir
        self.sample_list = [f for f in os.listdir(path_dir) if f.endswith('.pt')]
        self.trace_max, self.trace_min, self.log_max, self.log_min = self.dataset_norm()
        
        self.trace_mask = (self.trace_max != self.trace_min)
        self.log_mask = (self.log_max != self.log_min)

        # print(self.trace_max.shape, self.trace_mask.shape)
        self.trace_max = self.trace_max[self.trace_mask]
        self.trace_min = self.trace_min[self.trace_mask]
        self.log_max = self.log_max[self.log_mask]
        self.log_min= self.log_min[self.log_mask]

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.sample_list[idx]
        sample = torch.load(os.path.join(self.data_path,file_name))
        sample['timestamp'] = int(file_name.split('.')[0])

        sample['trace'] = torch.nan_to_num(sample['trace'], nan=0.0)
        sample['trace'] = sample['trace'][:, self.trace_mask]
        sample['log'] = sample['log'][:, self.log_mask]

        sample['trace'] = (sample['trace'] - self.trace_min) / (self.trace_max - self.trace_min)
        sample['log'] = (sample['log'] - self.log_min) / (self.log_max - self.log_min)

        return sample

    def get_timestamp(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return int(self.sample_list[idx].split('.')[0])
    
    def __len__(self):
        return len(self.sample_list)
    
    def dataset_norm(self):
        trace = None
        log = None
        for file in self.sample_list:
            sample = torch.load(os.path.join(self.data_path,file))
            sample['trace'] = torch.nan_to_num(sample['trace'], nan=0.0)
            trace = sample['trace'].reshape(-1, sample['trace'].shape[-1]) if trace is None else torch.concat([trace,sample['trace'].reshape(-1, sample['trace'].shape[-1])],dim=0)
            log = sample['log'].reshape(-1, sample['log'].shape[-1]) if log is None else torch.concat([log,sample['log'].reshape(-1, sample['log'].shape[-1])],dim=0)
        
        trace_max, _ = trace.max(dim=0)
        trace_min, _ = trace.min(dim=0)
        log_max, _ = log.max(dim=0)
        log_min, _ = log.min(dim=0)

        return trace_max, trace_min, log_max, log_min

def data_fliter(batch):
    m_batch = torch.stack([item['metric'] for item in  batch])
    t_batch = torch.stack([item['trace'] for item in  batch])
    l_batch = torch.stack([item['log'] for item in  batch])
    l_seq_batch = torch.stack([item['log_seq'] for item in  batch])
    label_batch = torch.stack([torch.tensor(1 - int((item['labels']==0).all())) for item in  batch])
    ts_batch = [item['timestamp'] for item in  batch]

    return m_batch.float(), t_batch.float(), l_batch.float(), l_seq_batch, label_batch, ts_batch