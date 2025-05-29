import torch
import torch.nn.functional as F

def SingleCenterLoss(z, center):
    mse_loss = F.mse_loss(z, center.unsqueeze(0), reduction='none')
    return mse_loss.sum(dim=1) 

def FocalCenterDistLossWeighted(z_list, center_list, weight_list, gamma=2):
    total_loss = None

    for (z, center, weight) in zip(z_list, center_list, weight_list):
        norm_center_loss = SingleCenterLoss(z, center) 
        norm_center_loss = norm_center_loss * weight 
        g = (z - center).abs().sum(dim=1) 
        g = torch.sigmoid(g.unsqueeze(1))
        norm_center_loss = norm_center_loss * (g**gamma)
        
        loss = norm_center_loss.unsqueeze(-1) 
        total_loss = loss if total_loss is None else torch.cat([total_loss,loss],dim=1)
    
    return total_loss.sum(dim=1).mean()

def ClusteringAnomalyScore(z_list, center_list, weight_list):
    total_loss = None

    for (z, center, weight) in zip(z_list, center_list, weight_list):
        norm_center_loss = F.mse_loss(z, center.unsqueeze(0), reduction='none').sum(dim=1)
        norm_center_loss = norm_center_loss * weight 
        loss = norm_center_loss.unsqueeze(-1)  
        total_loss = loss if total_loss is None else torch.cat([total_loss,loss],dim=1)
    return total_loss.mean(dim=1) 