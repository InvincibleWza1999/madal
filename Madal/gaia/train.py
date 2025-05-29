import torch
import pandas as pd
import sys
import numpy as np
from utils.SavedDataset import SavedDataset, data_fliter
# from utils.get_data import get_log_template_df
from utils.log_embedding import template2vec, log2vec, SVD_dimension_reduction
import argparse
import torch, time
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from network.mymodel import SimilarityModule, MultiReconstructNet, MultiModalDetection
from utils.loss import ClusteringAnomalyScore, FocalCenterDistLossWeighted
import copy
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.metrics import precision_recall_curve, confusion_matrix
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from transformers import get_cosine_schedule_with_warmup
import random

def prepare_data(modal_1, modal_2, modal3_fixed, modal4_fixed, device, shifts = 3):

    fixed_modal_3 = copy.deepcopy(modal3_fixed)
    fixed_modal_4 = copy.deepcopy(modal4_fixed)
    matched_modal_1 = copy.deepcopy(modal_1)
    unmatched_modal_1 = copy.deepcopy(modal_1).roll(shifts = shifts, dims=0)

    matched_modal_2 = copy.deepcopy(modal_2)
    unmatched_modal_2 = copy.deepcopy(modal_2).roll(shifts = shifts, dims=0)
    return [fixed_modal_3.cuda(), fixed_modal_4.cuda()], [matched_modal_1.cuda(), unmatched_modal_1.cuda()], [matched_modal_2.cuda(), unmatched_modal_2.cuda()]



def optimize_weights_grid(s1, s2, y):
    best_f1 = -1
    best_w1, best_w2 = 0, 1
    best_th = 0
    best_p, best_r = 0, 0
    for w1 in np.arange(0, 1.001, 0.001 ):
        w2 = 1 - w1
        weighted_score = w1 * s1 + w2 * s2
        precision, recall, ths = precision_recall_curve(y, weighted_score, pos_label=1)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        current_f1 = f1[np.argmax(f1)]
        current_p = precision[np.argmax(f1)]
        current_r = recall[np.argmax(f1)]
        current_th = ths[np.argmax(f1)]

        y_pred = (weighted_score >= current_th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_w1, best_w2 = w1, w2
            best_th = current_th
            best_p, best_r = current_p, current_r
            best_tp, best_fp, best_fn = tp, fp, fn
    
    return best_w1, best_w2, best_f1, best_p, best_r, best_th, best_tp, best_fp, best_fn, len(y_pred)


def model_eval(sim_model, rec_model, model, test_dataloader, center_list, embeddings):
    sim_model.eval()
    rec_model.eval()
    model.eval()

    val_true, val_pred = [], []
    occ_as_list = []
    rec_as_list  = []
    mse_loss = torch.nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for step, (batch_m, batch_t, batch_l, batch_l_seq, batch_label, ts) in enumerate(test_dataloader):
            batch_l_seq = log2vec(batch_l_seq, embeddings).float()

            batch_m, batch_t, batch_l, batch_l_seq, batch_label = \
                batch_m.cuda(), batch_t.cuda(), batch_l.cuda(), batch_l_seq.cuda(), batch_label.cuda()

            metric_aligned, log_aligned, trace_aligned = sim_model(batch_m, batch_t, batch_l, batch_l_seq) 
            attention_score, skl_score, m, l, t, correlation = \
                model(batch_m, batch_t, batch_l, batch_l_seq, 
                      metric_aligned, trace_aligned, log_aligned)
            
            m = m.squeeze()
            l = l.squeeze()
            t = t.squeeze()
            correlation = correlation.squeeze()

            occ_as = ClusteringAnomalyScore([m, l, t, correlation], center_list,
                               [attention_score[:,0],attention_score[:,1],attention_score[:,2],attention_score[:,3]])
            
            rec_m, rec_t, rec_l, fused_rec_m, fused_rec_t, fused_rec_l = rec_model(correlation, m, t, l)
            
            rec_as =  mse_loss(rec_m, batch_m).sum(dim=(1,2)) * attention_score[:,0] + \
                    mse_loss(rec_t, batch_t).sum(dim=(1,2)) * attention_score[:,2] + \
                    mse_loss(rec_l, batch_l).sum(dim=(1,2))* attention_score[:,1] + \
                    (mse_loss(fused_rec_m, batch_m).sum(dim=(1,2)) + mse_loss(fused_rec_t, batch_t).sum(dim=(1,2)) + \
                        mse_loss(fused_rec_l, batch_l).sum(dim=(1,2)))* attention_score[:,3]

            final_as = occ_as + rec_as
            
            val_pred.extend(final_as.squeeze().cpu().numpy().tolist())
            val_true.extend(batch_label.squeeze().cpu().numpy().tolist())

            occ_as_list.extend(occ_as.squeeze().cpu().numpy().tolist())
            rec_as_list.extend(rec_as.squeeze().cpu().numpy().tolist())
    
    best_w1, best_w2, best_f1, best_p, best_r, best_th, best_tp, best_fp, best_fn, total_len = optimize_weights_grid(np.array(occ_as_list),np.array(rec_as_list), np.array(val_true))

    return best_w1, best_w2, best_f1, best_p, best_r, best_th, best_tp, best_fp, best_fn, total_len


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

def main():

    service = 'dbservice1'    
    batch_size = 256
    epoch_num = 100
    weight_decay = 1e-3
    input_dim = 128
    beta = 0.5
    log_seq_len =  2048
    gat_setting =  [4, 32]
    lr = 5e-5
    seed = 174


    set_seed(seed)

    gat_out_channel = gat_setting[1]
    gat_head = gat_setting[0]

    metric_feature_dim = 84
    trace_feature_dim = 9
    log_feature_dim = 26
    window_size = 25  
    metric_nhead = 4
    trace_nhead = 4
    log_nhead = 8
    param_center=True
    
    log_template_df = pd.read_csv(f'log_template.csv')
    step = 25
    interval = 30000
    
    assert gat_out_channel*gat_head == input_dim, 'unmatched dim'
    
    dataset = SavedDataset(f"data/")
    normal_set = [item for item in dataset if (item['labels']==0).all()]

    normal_train_set, normal_test_set, y_train, y_test = train_test_split(
        normal_set, [0 for _ in range(len(normal_set))], 
        test_size = 0.2,           
        random_state = 42,         
    )
    
    anomaly_test_set = [item for item in dataset if (item['labels'] !=0).any()]
    train_loader = DataLoader(dataset=normal_train_set, batch_size = batch_size, shuffle=True, collate_fn = data_fliter, drop_last=True, num_workers=4)
    test_loader = DataLoader(dataset=anomaly_test_set + normal_test_set, batch_size = 32, shuffle=False, collate_fn = data_fliter,  drop_last=True, num_workers=4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for step, (batch_m, batch_t, batch_l, batch_l_seq, batch_label, ts) in enumerate(train_loader):
        metric_feature_dim = batch_m.shape[-1]
        trace_feature_dim = batch_t.shape[-1]
        log_feature_dim = batch_l.shape[-1]
        break

    log_embedding_model = SentenceTransformer('all-mpnet-base-v2',device=device)
    for param in log_embedding_model.parameters():
        param.requires_grad = False

    embeddings = template2vec(log_template_df, log_embedding_model)   
    embeddings, log_seq_dim = SVD_dimension_reduction(embeddings, 64)

    model = MultiModalDetection(metric_feature_dim, trace_feature_dim, log_feature_dim, 
                            log_seq_dim, device, input_dim, gat_head, metric_nhead, trace_nhead, 
                            log_nhead, gat_out_channel, window_size, log_seq_len, param_center)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    model.to(device)

    sim_model = SimilarityModule(metric_feature_dim, trace_feature_dim, log_feature_dim, 
                            log_seq_dim, device, input_dim, gat_head, metric_nhead, trace_nhead, 
                            log_nhead, gat_out_channel, window_size, log_seq_len)
    sim_model = torch.nn.DataParallel(sim_model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    sim_model.to(device)

    rec_model = MultiReconstructNet(input_dim, window_size, metric_feature_dim, log_seq_dim,log_feature_dim, trace_feature_dim)
    rec_model = torch.nn.DataParallel(rec_model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    rec_model.to(device)

    optim_task_similarity = torch.optim.AdamW(
        sim_model.parameters(), 
        lr=lr, weight_decay = weight_decay) 

    optim_task_AD = torch.optim.AdamW(
        params=list(model.parameters()) + list(rec_model.parameters()),
        lr=lr, weight_decay = weight_decay)
    
    total_steps = len(train_loader) * epoch_num
    warmup_steps = int(0.1 * total_steps) 

    scheduler = get_cosine_schedule_with_warmup(
        optim_task_AD,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
        )

    sim_loss = torch.nn.CosineEmbeddingLoss(margin=0.1)
    kl_loss =  torch.nn.KLDivLoss(reduction='batchmean')
    mse_loss = torch.nn.MSELoss(reduction='none')
    min_loss = np.inf
    i = 0
    best_f1 = 0.0
    best_p = 0.0
    best_r = 0.0
    best_th = 0.0
    best_w1 = 0.0
    best_w2 = 0.0
    best_tp = 0
    best_fp = 0
    best_fn = 0
    total_len = 0

    for epoch in range(epoch_num):
        start = time.time()
        sim_model.train()
        model.train()
        rec_model.train()
        print("***** Running training epoch {} *****".format(epoch+1))

        train_loss_ad = 0.0
        train_loss_sim = 0.0
        train_loss_skl = 0.0
        train_loss_rec = 0.0
        train_loss_occ = 0.0

        for step, (batch_m, batch_t, batch_l, batch_l_seq, batch_label, ts) in enumerate(train_loader):
            batch_l_seq = log2vec(batch_l_seq, embeddings).float()  
            
            #task 1: CL
            [fixed_batch_l, fixed_batch_l_seq], [matched_metric, unmatched_metric], [matched_trace, unmatched_trace] = prepare_data(batch_m, batch_t, batch_l, batch_l_seq, device)
            metric_aligned_match, log_aligned_match, trace_aligned_match = sim_model(matched_metric, matched_trace, fixed_batch_l, fixed_batch_l_seq)
            metric_aligned_unmatch, log_aligned_unmatch, trace_aligned_unmatch = sim_model(unmatched_metric, unmatched_trace, fixed_batch_l, fixed_batch_l_seq)
            
            metric_aligned = torch.cat([metric_aligned_match, metric_aligned_unmatch], dim=0)
            trace_aligned = torch.cat([trace_aligned_match, trace_aligned_unmatch], dim=0)
            log_aligned =  torch.cat([log_aligned_match, log_aligned_unmatch], dim=0)

            similarity_label = torch.cat([torch.ones(metric_aligned_match.shape[0]), -1 * torch.ones(metric_aligned_unmatch.shape[0])], dim=0).to(device)          
            loss_similarity = sim_loss(metric_aligned, log_aligned, similarity_label) + sim_loss(trace_aligned, log_aligned, similarity_label)
            
            optim_task_similarity.zero_grad()
            loss_similarity.backward()
            torch.nn.utils.clip_grad_norm_(sim_model.parameters(), max_norm=1.0)
            optim_task_similarity.step()


            #task2: AD
            metric_aligned_match, log_aligned_match, trace_aligned_match = sim_model(matched_metric, matched_trace, fixed_batch_l, fixed_batch_l_seq)
            attention_score, skl_score, m, l, t, correlation = model(matched_metric, matched_trace, fixed_batch_l, fixed_batch_l_seq, metric_aligned_match, trace_aligned_match, log_aligned_match)
            skl_loss = kl_loss(attention_score, skl_score) / 2 +  kl_loss(skl_score, attention_score) / 2           

            
            m = m.squeeze()
            l = l.squeeze()
            t = t.squeeze()
            correlation = correlation.squeeze()

            occ_loss = FocalCenterDistLossWeighted([m, l, t, correlation], 
                                [model.module.m_center,  model.module.l_center, model.module.t_center, model.module.fused_center],
                                [attention_score[:,0],attention_score[:,1],attention_score[:,2],attention_score[:,3]])

            
            rec_m, rec_t, rec_l, fused_rec_m, fused_rec_t, fused_rec_l = rec_model(correlation, m, t, l)
    
            
            rec_loss = mse_loss(rec_m, matched_metric).sum(dim=(1,2)) * attention_score[:,0] + \
                        mse_loss(rec_t, matched_trace).sum(dim=(1,2)) * attention_score[:,2] + \
                        mse_loss(rec_l, fixed_batch_l).sum(dim=(1,2))* attention_score[:,1] + \
                        (mse_loss(fused_rec_m, matched_metric).sum(dim=(1,2)) + mse_loss(fused_rec_t, matched_trace).sum(dim=(1,2)) + \
                            mse_loss(fused_rec_l, fixed_batch_l).sum(dim=(1,2)))* attention_score[:,3]
        
            
            rec_loss = rec_loss.mean()

            total_loss = skl_loss + beta * occ_loss + (1-beta) * rec_loss
            optim_task_AD.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(rec_model.parameters(), max_norm=1.0)
            optim_task_AD.step()
            scheduler.step()

            train_loss_ad += total_loss.item()
            train_loss_rec += rec_loss.item()
            train_loss_occ += occ_loss.item()
            train_loss_skl += skl_loss.item()
            train_loss_sim += loss_similarity.item()

            print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(
                    epoch+1, step+1, len(train_loader), train_loss_ad/(step+1), time.time() - start))
            print("Sim Loss: {:.4f} | Skl loss: {:.4f} | Occ Loss {:.4f} | Rec Loss {:.4f}".format(
                    train_loss_sim/(step+1), train_loss_skl/(step+1), train_loss_occ/(step+1), train_loss_rec/(step+1)))
       
        center_list =  [model.module.m_center,  model.module.l_center, model.module.t_center, model.module.fused_center]
        try:
            w1, w2, f1, p, r, th, tp, fp, fn, current_len = model_eval(sim_model, rec_model, model, test_loader, center_list, embeddings)
        except ValueError:
            print(f"encouter value error at epoch {epoch}")
            break
    

        if f1 >= best_f1:
            best_f1 = f1
            best_p = p
            best_r = r
            best_w1 = w1
            best_w2 = w2
            best_th = th
            best_tp = tp
            best_fp = fp
            best_fn = fn 
            total_len = current_len
            
                    
        print("Epoch {}, training time: {:.4f}, evaluating time: {:.4f}".format(epoch+1, end, end_1))
        print("Epoch {}, f1: {:.4f}, pre: {:.4f}, recall: {:.4f}".format(epoch+1, f1, p, r))

    print('*******************************************************************')
    print("model params: input_dim : {}, beta: {}, log_seq_len: {}, gat_head: {}, gat_channel: {}, lr: {}, seed: {} \n".format(input_dim, beta, log_seq_len, gat_head, gat_out_channel, lr, seed))
    print("best f1: {:.4f} | best pre: {:.4f} | best recall: {:.4f} | best weights: {:.4f}, {:.4f} | best threshold: {:.4f}".format(best_f1, best_p, best_r, best_w1, best_w2, best_th))
    print('*******************************************************************')

if __name__ == "__main__":
    main()
