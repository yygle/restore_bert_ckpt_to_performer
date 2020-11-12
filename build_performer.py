# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import os
from performer_pytorch.wrappers import ClassificationWrapper
# from performer_pytorch import PerformerLM
from performer_pytorch_v2 import PerformerLM
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_model(model, check_point_path, device=None, state_only=False, prefix=None):
    checkpoint = torch.load(check_point_path, map_location=torch.device("cpu"))
    if state_only:
        checkpoint = checkpoint
    else:
        checkpoint = checkpoint["model"]
    own_state = model.state_dict()
    # for k, v in own_state.items():
    #     print(k, v.size())
    for name, param in checkpoint.items():
        if prefix:
            name = '.'.join(prefix) + '.' + name
        if name not in own_state:
            print('{} not found'.format(name))
            continue
        if param.data.shape != own_state[name].shape:
            print('{} not found different shape'.format(name))
            continue
        print('{} loaded!'.format(name))
        param = param.data
        own_state[name].copy_(param)
    return model, checkpoint

PERFORMER_CONFIG_V1 = {
    'num_tokens': 45126,
    'dim': 512,
    'depth': 6,
    'max_seq_len': 128,
    'heads': 8,
    'causal': True,
    'reversible': True,
    'nb_features': 256,
    'use_scalenorm': True,
    'tie_embedding': True
}

PERFORMER_CONFIG_BERT = {
    'num_tokens': 21128,
    'dim': 768,
    'depth': 12,
    'max_seq_len': 512,
    'heads': 12,
    'causal': True,
    'reversible': True,
    'nb_features': 256,
    'use_scalenorm': True,
    # 'tie_embedding': True,
    # 'remap_vocab': False
}

# model = PerformerLM(**PERFORMER_CONFIG_V1)
model = PerformerLM(**PERFORMER_CONFIG_BERT)
model = ClassificationWrapper(model, num_labels=2)
out_performer_model = './saved_model/converted_performer_from_roberta.bin'
out_performer_model3 = './saved_model/converted_performer_from_roberta2.bin'
# model, _ = load_model(model, out_performer_model, state_only=True, prefix=['encoder'])
model, _ = load_model(model, out_performer_model3, state_only=True, prefix=['encoder'])
