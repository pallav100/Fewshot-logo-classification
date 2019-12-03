from protonet import ProtoNet
from prototypical_loss import my_prototypical_loss as loss_fn
import numpy as np
import torch
from PIL import Image
import os
import pickle
def get_embd(x, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
#    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
#     x, y = x.to(device), y.to(device)
    out = model(x)
    return (out)
model =ProtoNet()
def load_img(path):
    x = Image.open(path).convert('RGB')
    x = x.resize((28, 28))
    shape = 3, x.size[0], x.size[1]
    x = np.array(x, np.float32, copy=False)
    x = 1.0 - torch.from_numpy(x)
    x = x.transpose(0, 1).contiguous().view(shape)
    return x
img = load_img('/home/pallav_soni/dumm.jpeg')
img = torch.unsqueeze(img, 0)
#print(img.size())
embd = get_embd(img,model)
means_path = '/home/pallav_soni/pro/model/prev_latest_means.pt'

log_query = loss_fn(embd,means_path)
print(log_query[0].size())
with open('/home/pallav_soni/pro/model/classes.pkl', 'rb') as f:
	classlist = pickle.load(f)
#inv_classlist = {v: k for k, v in classlist.items()}
sorted_x = sorted(classlist.items(), key=lambda kv: kv[1])
for item  in sorted_x:
    print("{} : {}".format(item[1],item[0]))
