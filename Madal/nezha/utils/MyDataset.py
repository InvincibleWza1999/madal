import torch
# from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os, math
import sys
import pandas as pd
import os
import numpy as np

def get_single_service_metric_data(metric_df:pd.DataFrame, window_size, start_time, interval = 60000, complete_ratio = 0.6, all_na_ratio = 0.05):

    end_time = start_time + interval * window_size
    metric_sample = None
    complete_metric_num = 0
    all_na_metric = []

    metric_split_data =  metric_df[(metric_df['timestamp']>=start_time) & (metric_df['timestamp']<end_time)]
    for col in metric_split_data.columns:
        if col != 'timestamp':
            temp = metric_split_data[col].dropna().reset_index(drop=True)
            # print(len(temp))

            if len(temp) >= window_size:
                complete_metric_num += 1
            
            if len(temp) == 0:
                all_na_metric.append(col)
                
            if metric_sample is None:
                metric_sample = temp
            else:
                metric_sample = pd.concat([metric_sample, temp],axis=1)

    if len(all_na_metric) >= all_na_ratio * (metric_df.shape[1] - 1):  
        print(f"too many empty metrics at start time {start_time}: {all_na_metric}")
        return None
    
    if complete_metric_num >= complete_ratio * (metric_df.shape[1] - 1):
        metric_sample.reset_index(inplace=True,drop=True)
        metric_sample = metric_sample[0:window_size].interpolate()
        metric_sample = metric_sample.fillna(0)
        return np.array(metric_sample)
    else:
        print(f"too many metrics with missing value at start time {start_time}")
        return None


def get_single_service_trace_data(trace_df:pd.DataFrame, window_size, start_time, interval = 30000):
    # trace_df = trace_df.drop_duplicates(['timestamp'])
    end_time = start_time + interval * window_size

    trace_split_data = trace_df[(trace_df['timestamp']>=start_time) & (trace_df['timestamp']<end_time)]
    trace_split_data = trace_split_data.sort_values(by = 'timestamp')
    # trace_df['duration'] = trace_df['start_time'] - trace_df['end_time']   #ms

    if len(trace_split_data) != 0:
        span_series = []
        for w in range(start_time, end_time, interval):
            trace_window = trace_split_data[(trace_split_data['timestamp']>=w) & (trace_split_data['timestamp']<w+interval)]
            if len(trace_window) != 0:
                span_num = len(trace_window)
                span_mean = trace_window['duration'].mean()
                span_std = trace_window['duration'].std()
                span_max = trace_window['duration'].max()
                span_min = trace_window['duration'].min()
                span_25 = trace_window['duration'].quantile(0.25)
                span_50 = trace_window['duration'].quantile(0.5)
                span_75 = trace_window['duration'].quantile(0.75)
                # span_200_ratio = len(trace_window[trace_window['status_code']==200]) / len(trace_window)
                span_info = [span_num, span_mean, span_std, span_max, span_min, span_25, span_50, span_75]
            else:
                span_info = [0 for _ in range(8)]

        
            span_series.append(span_info)
        return np.array(span_series)
    else:
        
        return np.zeros((window_size,8))


def get_single_service_log_data(log_df:pd.DataFrame, window_size, start_time, interval = 30000):
    end_time = start_time + interval * window_size

    log_split_data = log_df[(log_df['timestamp']>=start_time) & (log_df['timestamp']<end_time)]
    # log_split_data = log_split_data.sort_values(by = 'timestamp')
    
    unique_log_id = sorted(log_df['ID'].unique().tolist())
    # print(unique_log_id)
    if len(log_split_data) != 0:
        log_series = []
        for w in range(start_time, end_time, interval):
            log_window = log_split_data[(log_split_data['timestamp']>=w) & (log_split_data['timestamp']<w+interval)]
            if len(log_window) != 0:
                counts = log_window['ID'].value_counts()
                counts = counts.reindex(unique_log_id, fill_value=0)
            else:
                counts = [0 for _ in range(len(unique_log_id))]
            log_series.append(counts)
        return np.array(log_series)
    
    else:
        return np.zeros((window_size, len(unique_log_id)))
    

def get_single_service_log_seq(log_df:pd.DataFrame, window_size, start_time, interval = 30000, seq_len = 256):
    end_time = start_time + interval * window_size

    log_split_data = log_df[(log_df['timestamp']>=start_time) & (log_df['timestamp']<end_time)]
    # log_split_data = log_split_data.sort_values(by = 'timestamp')
    # print(unique_log_id)
    if len(log_split_data) != 0:
        log_seqs = []
        for w in range(start_time, end_time, interval):
            log_window = log_split_data[(log_split_data['timestamp']>=w) & (log_split_data['timestamp']<w+interval)]['ID'].to_list()

            if len(log_window) > seq_len:
                log_window = log_window[:seq_len]
            else:
                log_window = log_window + [0 for _ in range(seq_len - len(log_window))]
            log_seqs.append(log_window)
        return np.array(log_seqs)
    
    else:
        return np.array([[0 for i in range(seq_len)] for j in range(window_size)])


def get_single_service_log_seq_complete(log_df:pd.DataFrame, window_size, start_time, interval = 30000, seq_len = 2048):
    end_time = start_time + interval * window_size

    log_split_data = log_df[(log_df['timestamp']>=start_time) & (log_df['timestamp']<end_time)]['ID'].to_list()

    if len(log_split_data) != 0:
        
        if len(log_split_data) > seq_len:
            log_split_data = log_split_data[:seq_len]
        else:
            log_split_data = log_split_data + [0 for _ in range(seq_len - len(log_split_data))]
        return np.array(log_split_data)
    
    else:
        return np.array([0 for _ in range(seq_len)])
    

def get_single_service_label_data(label_df:pd.DataFrame, window_size, start_time, interval = 30000):
    end_time = start_time + interval * window_size
    label_split_data = label_df[(label_df['timestamp']>=start_time) & (label_df['timestamp']<end_time)]

    if len(label_split_data) != 0:
        return np.array(label_split_data['label'])
    else:
        return None
    


class Mydataset(Dataset):
    def __init__(self, window_size, step_size, interval, log_seq_len, metric_df, trace_df,log_df,label_df):
        self.window_size = window_size
        self.step_size = step_size
        self.interval = interval
        self.metric_df = metric_df
        self.trace_df = trace_df
        self.log_df = log_df
        self.label_df = label_df
        self.log_seq_len = log_seq_len

        self.start_time = max(label_df['timestamp'].min(),metric_df['timestamp'].min(),trace_df['timestamp'].min(),log_df['timestamp'].min())
        self.end_time = min(label_df['timestamp'].max(),metric_df['timestamp'].max(),trace_df['timestamp'].max(),log_df['timestamp'].max())
        
    def __getitem__(self, i):
        m, t, l, l_seq ,label = get_single_service_metric_data(self.metric_df,self.window_size, self.start_time + i * self.interval * self.step_size, self.interval),  \
                            get_single_service_trace_data(self.trace_df,self.window_size,self.start_time + i * self.interval * self.step_size, self.interval),  \
                                get_single_service_log_data(self.log_df,self.window_size,self.start_time + i * self.interval * self.step_size, self.interval), \
                                    get_single_service_log_seq_complete(self.log_df,self.window_size,self.start_time + i * self.interval * self.step_size, self.interval, self.log_seq_len), \
                                        get_single_service_label_data(self.label_df, self.window_size, self.start_time + i * self.interval * self.step_size, self.interval)

        if m is None or label is None:
            return None
        else:
            # return torch.from_numpy(m), torch.from_numpy(t), torch.from_numpy(l), torch.from_numpy(l_seq),torch.tensor( 1 - int((label==0).all()))
            return torch.from_numpy(m), torch.from_numpy(t), torch.from_numpy(l), torch.from_numpy(l_seq),torch.tensor(label)
        # label 0 if normal else 1

    def __len__(self):
        return max(0, math.ceil((self.end_time - self.start_time) / (self.step_size * self.interval)))

    def get_timestamp(self,i):
        return int(self.start_time + i * self.interval * self.step_size)


def data_fliter(batch):
    valid_batch = [item for item in batch if item is not None]

    if not valid_batch:
        return [], [], [], []
    
    m_batch = torch.stack([item[0] for item in valid_batch])
    t_batch = torch.stack([item[1] for item in valid_batch])
    l_batch = torch.stack([item[2] for item in valid_batch])
    l_seq_batch = torch.stack([item[3] for item in valid_batch])
    label_batch = torch.stack([item[4] for item in valid_batch])
    
    return m_batch, t_batch, l_batch, l_seq_batch, label_batch