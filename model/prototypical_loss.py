# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support,i_ep,t_ep, mode='val', nit = 1000000000, niit = 10000000000,final = 0):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    traintensor = '/home/pallav_soni/pro/model/train_means.pt'
    latest_tensor = '/home/pallav_soni/pro/model/latest_means.pt'
    prev_latest_tensor = '/home/pallav_soni/pro/model/prev_latest_means.pt'

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
 #   print(classes)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
#    print(support_idxs)
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    if(i_ep==t_ep and nit==niit and mode=='train' ):
        torch.save(prototypes, traintensor)
        torch.save(prototypes, prev_latest_tensor)
 #       print('saved training tensor')
#        print(prototypes.size())
    if(mode=='test'):
       a = torch.load(prev_latest_tensor)
       b = torch.cat([a,prototypes],dim=0)
       if(nit==niit and i_ep==t_ep):
           print('saving to lat')
           torch.save(b,latest_tensor)
           if(final):
             print('saving final few shot model')
             torch.save(b,prev_latest_tensor)
    query_samples = input.to('cpu')[query_idxs]
    if(mode=='test'):
        dists = euclidean_dist(query_samples, b)

    else:
        dists =euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
 #   if(t_ep == i_ep and mode =='test'):
#           print(log_p_y.max(2))
    if(mode=='test'):
           target_inds = torch.arange(0, n_classes)
           target_inds = target_inds.view(n_classes, 1, 1)
           target_inds = target_inds.expand(n_classes, n_query, 1).long()
           t = torch.tensor((),dtype = torch.long)
           t = t.new_full((n_classes,n_query,1), a.size()[0])
           target_inds = torch.add(target_inds,t)
    else:
        target_inds = torch.arange(0, n_classes)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()
#    if(t_ep == i_ep and mode =='test'):
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val
def my_prototypical_loss(query,path):
	avgs = torch.load(path)
	dists = euclidean_dist(query,avgs)
	log_p_y = F.log_softmax(-dists, dim=1).view(1, 1, -1)
	print(log_p_y)
	lbl = log_p_y.max(2)
	return(lbl)
