# -*- coding:utf-8 -*-
import collections

from ptbert import *


def analysis_model(path='./data/cache/model.bin', analysis_paras=False):
    print('=' * 25 + path + '=' * 25)
    model = torch.load(path)
    try:
        own_state = model.state_dict()
    except:
        own_state = model
    paras_num = 0
    for k, v in own_state.items():
        print(k, '\t', v.size())
        if analysis_paras:
            paras_num += np.prod([*v.size()])
    if analysis_paras:
        print('total parameters: {:,}'.format(paras_num))


def load_model(model, check_point_path, device=None):
    checkpoint = torch.load(check_point_path, map_location=torch.device("cpu"))
    own_state = model.state_dict()
    for name, param in checkpoint["model"].items():
        if name not in own_state:
            print('{} not found'.format(name))
            continue
        if param.data.shape != own_state[name].shape:
            print('{} not found different shape'.format(name))
            continue
        print('{} loaded'.format(name))
    param = param.data
    own_state[name].copy_(param)
    return model, checkpoint


def convert_bert_ckpt_to_performer(bert_path, performer_path, layers=12):
    if not os.path.exists(bert_path):
        raise Exception('Please check your bert model')
    model = torch.load(bert_path)
    try:
        bert_state = model.state_dict()
    except:
        bert_state = model
    # print(type(bert_state))
    # for k, v in bert_state.items():
    #     print(k, '\t', v.size())
    performer_state = collections.OrderedDict()
    performer_state['token_emb.weight'] = bert_state['bert.embeddings.word_embeddings.weight']
    performer_state['pos_emb.weight'] = bert_state['bert.embeddings.position_embeddings.weight']
    for layer in range(layers):
        q_w = bert_state['bert.encoder.layer.' + str(layer) +'.attention.self.query.weight']
        k_w = bert_state['bert.encoder.layer.' + str(layer) +'.attention.self.key.weight']
        v_w = bert_state['bert.encoder.layer.' + str(layer) +'.attention.self.value.weight']
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        # q_b = bert_state['bert.encoder.layer.' + str(layer) + '.attention.self.query.bias']
        # k_b = bert_state['bert.encoder.layer.' + str(layer) + '.attention.self.key.bias']
        # v_b = bert_state['bert.encoder.layer.' + str(layer) + '.attention.self.value.bias']
        # qkv_b = torch.cat([q_b, k_b, v_b], dim=0)
        # skip g
        performer_state['performer.net.blocks.' + str(layer) +'.f.net.fn.to_qkv.weight'] = qkv_w
        performer_state['performer.net.blocks.' + str(layer) +'.f.net.fn.to_out.weight'] = \
            bert_state['bert.encoder.layer.' + str(layer) +'.attention.output.dense.weight']
        performer_state['performer.net.blocks.' + str(layer) + '.f.net.fn.to_out.bias'] = \
            bert_state['bert.encoder.layer.' + str(layer) + '.attention.output.dense.bias']
        # skip g
        performer_state['performer.net.blocks.' + str(layer) + '.g.net.fn.fn.net.0.weight'] = \
            bert_state['bert.encoder.layer.' + str(layer) + '.intermediate.dense.weight']
        performer_state['performer.net.blocks.' + str(layer) + '.g.net.fn.fn.net.0.bias'] = \
            bert_state['bert.encoder.layer.' + str(layer) + '.intermediate.dense.bias']
        performer_state['performer.net.blocks.' + str(layer) + '.g.net.fn.fn.net.2.weight'] = \
            bert_state['bert.encoder.layer.' + str(layer) + '.output.dense.weight']
        performer_state['performer.net.blocks.' + str(layer) + '.g.net.fn.fn.net.2.bias'] = \
            bert_state['bert.encoder.layer.' + str(layer) + '.output.dense.bias']
    torch.save(performer_state, performer_path)
    print('Successfully convert bert to performer!')
    print('Please check path: ', performer_path)

def convert_bert_ckpt_to_performer_v2(bert_path, performer_path, layers=12):
    if not os.path.exists(bert_path):
        raise Exception('Please check your bert model')
    model = torch.load(bert_path)
    try:
        bert_state = model.state_dict()
    except:
        bert_state = model
    # print(type(bert_state))
    # for k, v in bert_state.items():
    #     print(k, '\t', v.size())
    performer_state = collections.OrderedDict()
    performer_state['token_emb.weight'] = bert_state['bert.embeddings.word_embeddings.weight']
    performer_state['pos_emb.weight'] = bert_state['bert.embeddings.position_embeddings.weight']
    for layer in range(layers):
        performer_state['performer.net.blocks.' + str(layer) + '.f.net.fn.to_q.weight'] = \
            bert_state['bert.encoder.layer.' + str(layer) +'.attention.self.query.weight']
        performer_state['performer.net.blocks.' + str(layer) + '.f.net.fn.to_q.bias'] = \
            bert_state['bert.encoder.layer.' + str(layer) + '.attention.self.query.bias']
        performer_state['performer.net.blocks.' + str(layer) + '.f.net.fn.to_k.weight'] = \
            bert_state['bert.encoder.layer.' + str(layer) +'.attention.self.key.weight']
        performer_state['performer.net.blocks.' + str(layer) + '.f.net.fn.to_k.bias'] = \
            bert_state['bert.encoder.layer.' + str(layer) + '.attention.self.key.bias']
        performer_state['performer.net.blocks.' + str(layer) + '.f.net.fn.to_v.weight'] = \
            bert_state['bert.encoder.layer.' + str(layer) +'.attention.self.value.weight']
        performer_state['performer.net.blocks.' + str(layer) + '.f.net.fn.to_v.bias'] = \
            bert_state['bert.encoder.layer.' + str(layer) + '.attention.self.value.bias']
        # skip g
        performer_state['performer.net.blocks.' + str(layer) +'.f.net.fn.to_out.weight'] = \
            bert_state['bert.encoder.layer.' + str(layer) +'.attention.output.dense.weight']
        performer_state['performer.net.blocks.' + str(layer) + '.f.net.fn.to_out.bias'] = \
            bert_state['bert.encoder.layer.' + str(layer) + '.attention.output.dense.bias']
        # skip g
        performer_state['performer.net.blocks.' + str(layer) + '.g.net.fn.fn.w1.weight'] = \
            bert_state['bert.encoder.layer.' + str(layer) + '.intermediate.dense.weight']
        performer_state['performer.net.blocks.' + str(layer) + '.g.net.fn.fn.w1.bias'] = \
            bert_state['bert.encoder.layer.' + str(layer) + '.intermediate.dense.bias']
        performer_state['performer.net.blocks.' + str(layer) + '.g.net.fn.fn.w2.weight'] = \
            bert_state['bert.encoder.layer.' + str(layer) + '.output.dense.weight']
        performer_state['performer.net.blocks.' + str(layer) + '.g.net.fn.fn.w2.bias'] = \
            bert_state['bert.encoder.layer.' + str(layer) + '.output.dense.bias']
    torch.save(performer_state, performer_path)
    print('Successfully convert bert to performer!')
    print('Please check path: ', performer_path)



if __name__ == '__main__':
    bert_model = './data/cache/model.bin'
    roberta_model = '/data5/yangyuguang/pro63/CLUE-master/classifier_pytorch/' \
                    'prev_trained_model/roberta_base_zh_torch/roberta_model.bin'
    performer_model = './saved_model/performer.bin'
    out_performer_model = './saved_model/converted_performer_from_bert.bin'
    out_performer_model2 = './saved_model/converted_performer_from_roberta.bin'
    out_performer_model3 = './saved_model/converted_performer_from_roberta2.bin'
    # analysis_model(roberta_model, analysis_paras=True)
    # analysis_model(bert_model, analysis_paras=True)
    # analysis_model(performer_model, analysis_paras=True)
    # convert_bert_ckpt_to_performer(bert_model, out_performer_model, layers=12)
    # convert_bert_ckpt_to_performer(roberta_model, out_performer_model2, layers=12)
    convert_bert_ckpt_to_performer_v2(roberta_model, out_performer_model3, layers=12)
    # load_model()
