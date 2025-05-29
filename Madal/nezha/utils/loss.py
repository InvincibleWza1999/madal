import torch
import torch.nn.functional as F

def SingleCenterLoss(z, center):
    mse_loss = F.mse_loss(z, center.unsqueeze(0), reduction='none')
    return mse_loss.mean(dim=1)  #[batch_size]

def FocalCenterDistLoss(z_list, center_list, weight_list, gamma=2):
    total_loss = None

    for (z, center, weight) in zip(z_list, center_list, weight_list):
        norm_center_loss = SingleCenterLoss(z, center) 
        # weight_norm = weight_norm.mean(dim=1)
        # norm_center_loss = SingleCenterLoss(z_norm, center).sum(dim=1) * weight_norm
        norm_center_loss = norm_center_loss * weight 
        g = (z - center).abs().sum(dim=1)  #[batch_size]
        g = torch.sigmoid(g.unsqueeze(1))
        norm_center_loss = norm_center_loss * (g**gamma)
        
        loss = norm_center_loss.unsqueeze(-1)  #[batch_size,1]
        total_loss = loss if total_loss is None else torch.cat([total_loss,loss],dim=1)
    
    # print(total_loss.shape)
    return total_loss.sum(dim=1).mean()


def WeightCenterDistLoss(z_list, center_list, weight_list):
    total_loss = None

    for (z, center, weight) in zip(z_list, center_list, weight_list):
        norm_center_loss = SingleCenterLoss(z, center) 
        # weight_norm = weight_norm.mean(dim=1)
        # norm_center_loss = SingleCenterLoss(z_norm, center).sum(dim=1) * weight_norm
        norm_center_loss = norm_center_loss * weight         
        loss = norm_center_loss.sum(dim=1).unsqueeze(-1)  #[batch_size,1]
        total_loss = loss if total_loss is None else torch.cat([total_loss,loss],dim=1)
    
    # print(total_loss.shape)
    return total_loss.sum(dim=1).mean()


def CenterDistLoss(z_list, center_list, weight_list):
    total_loss = None

    for (z, center, weight) in zip(z_list, center_list, weight_list):
        norm_center_loss = SingleCenterLoss(z, center) 
        # weight_norm = weight_norm.mean(dim=1)
        # norm_center_loss = SingleCenterLoss(z_norm, center).sum(dim=1) * weight_norm    
        loss = norm_center_loss.sum(dim=1).unsqueeze(-1)  #[batch_size,1]
        total_loss = loss if total_loss is None else torch.cat([total_loss,loss],dim=1)
    
    # print(total_loss.shape)
    return total_loss.sum(dim=1).mean()


def ClusteringAnomalyScore(z_list, center_list, weight_list):
    total_loss = None

    for (z, center, weight) in zip(z_list, center_list, weight_list):
        norm_center_loss = SingleCenterLoss(z, center) 
        # weight_norm = weight_norm.mean(dim=1)
        # norm_center_loss = SingleCenterLoss(z_norm, center).sum(dim=1) * weight_norm
        norm_center_loss = norm_center_loss * weight 
        
        loss = norm_center_loss.unsqueeze(-1)  #[batch_size,1]
        total_loss = loss if total_loss is None else torch.cat([total_loss,loss],dim=1)
    
    # print(total_loss.shape)
    return total_loss.mean(dim=1)  #[bs]

def VAEReconstructLoss(x, rec_x, mu, log_var, weight=0.5):
    #mu, log_var: [batch_size, latent_dim]
    #x: [batch_size, seq_len, input_dim]
    reconstruction_loss = torch.mean(torch.square(x - rec_x).sum(dim=1))  #TODO
    latent_loss = torch.mean(0.5*(log_var.exp() + mu.pow(2) - log_var).sum(dim=1))
    return reconstruction_loss + weight * latent_loss


# def FocalCenterDistLoss(z_list, center_list, weight_list, label:torch.Tensor):
#     total_loss = None
#     if (label==1).sum() != 0:
#         gamma = int(torch.log2((label==0).sum() / (label==1).sum()))
#     else:
#         gamma = 2

#     for (z, center, weight) in zip(z_list, center_list, weight_list):
#         z_norm = z[label==0]
#         weight_norm = weight[label==0]
#         norm_center_loss = SingleCenterLoss(z_norm, center) * weight_norm 
#         # weight_norm = weight_norm.mean(dim=1)
#         # norm_center_loss = SingleCenterLoss(z_norm, center).sum(dim=1) * weight_norm 
#         g = (z_norm - center).abs().sum(dim=1)  #[batch_size]
#         g = torch.sigmoid(g.unsqueeze(1))
#         norm_center_loss = norm_center_loss * (g**gamma)

#         z_ab = z[label==1]
#         weight_ab = weight[label==1]
#         ab_center_loss = weight_ab / SingleCenterLoss(z_ab, center) 
#         g = (z_ab - center).abs().sum(dim=1)  #[batch_size]
#         g = torch.sigmoid(g.unsqueeze(1))
#         ab_center_loss = ab_center_loss * (g**(-gamma))        
        
#         loss = torch.cat([norm_center_loss,ab_center_loss],dim=0).sum(dim=1).unsqueeze(-1)  #[batch_size,1]
#         total_loss = loss if total_loss is None else torch.cat([total_loss,loss],dim=1)
    
#     return total_loss.sum(dim=1).mean()




def FocalClassificationLoss(input, label, alpha = 0.75):
    BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(input, label, reduction='none')
    p_t = torch.exp(-BCE_loss)

    if (label==1).sum() != 0:
        gamma = max(1,int(torch.log2((label==0).sum() / (label==1).sum())))
    else:
        gamma = 1

    alpha_ = torch.ones_like(label)
    alpha_[label==1] = alpha
    alpha_[label==0] = 1 - alpha


    focal_loss = alpha * (1 - p_t) ** gamma * BCE_loss
    return focal_loss.mean()