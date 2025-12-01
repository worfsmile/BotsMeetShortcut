import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CLUB_NCE(nn.Module):
    def __init__(self, emb_dim = 300):
        super(CLUB_NCE, self).__init__()
        lstm_hidden_dim = emb_dim//2
        
        self.F_func = nn.Sequential(nn.Linear(lstm_hidden_dim*4, lstm_hidden_dim*2),
                                    #nn.Dropout(p=0.2),
                                    nn.ReLU(),
                                    nn.Linear(lstm_hidden_dim*2, 1),
                                    nn.Softplus())
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))#

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        upper_bound = T0.mean() - T1.mean()
        
        return lower_bound, upper_bound

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    
    def __init__(self, emb_dim = 300):
        super(CLUB, self).__init__()
        lstm_hidden_dim = emb_dim

        self.p_mu = nn.Sequential(nn.Linear(lstm_hidden_dim, lstm_hidden_dim//2),
                                nn.ReLU(),
                                nn.Linear(lstm_hidden_dim//2, lstm_hidden_dim))
       
        self.p_logvar = nn.Sequential(nn.Linear(lstm_hidden_dim, lstm_hidden_dim//2),
                                       nn.ReLU(),
                                       nn.Linear(lstm_hidden_dim//2, lstm_hidden_dim),
                                       nn.Tanh())
        

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 
        
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        lld = (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

        return lld , upper_bound

class InfoNCE(nn.Module):
    def __init__(self, emb_dim = 300):
        super(InfoNCE, self).__init__()
        print('InfoNCE')
        lstm_hidden_dim = emb_dim//2
        
        self.F_func = nn.Sequential(nn.Linear(lstm_hidden_dim*4, lstm_hidden_dim*2),
                                    #nn.Dropout(p=0.2),
                                    nn.ReLU(),
                                    nn.Linear(lstm_hidden_dim*2, 1),
                                    nn.Softplus())
                                    
    def forward(self, x_samples, y_samples): 
        sample_size = y_samples.shape[0]
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))#
        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]
        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size))
        return lower_bound


def manifold_consistency_loss(original_emb, new_emb, temperature=0.05):
    def pairwise_cos_sim(x):
        x = F.normalize(x, p=2, dim=1)
        return torch.matmul(x, x.T) / temperature  # shape: [B, B]

    orig_sim = pairwise_cos_sim(original_emb)
    new_sim = pairwise_cos_sim(new_emb)

    orig_sim_log_softmax = F.log_softmax(orig_sim, dim=1)
    new_sim_softmax = F.softmax(new_sim, dim=1)
    kl_loss = F.kl_div(orig_sim_log_softmax, new_sim_softmax, reduction='batchmean')
    return kl_loss
