from transformers.trainer import Trainer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel
from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from .utils import trim_batch
import logging
import os

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    pass

dist_logger = logging.getLogger('dist')


def count_flops(model, inputs, name):
    flops = FlopCountAnalysis(model, inputs)
    total = flops.total()
    dist_logger.info('Flop of {}: {}'.format(name, total))
    return total

class ForgettingPredictionTrainer(Trainer):
    def __init__(self, **kwargs):
        self.base_model = kwargs.pop('base_model')
        self.base_trainer = kwargs.pop('base_trainer')
        super().__init__(**kwargs)

class ContrastiveHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ForgettingPredictionModel(nn.Module):
    def __init__(self, config, tokenizer, helper, init_model=True):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.use_head = config.fpd.use_head
        self.normalize = config.fpd.normalize
        self.temp = config.fpd.temp
        self.head = None
        self.helper = helper

        if init_model:
            self.lm = AutoModelForSeq2SeqLM.from_pretrained(config.fpd.model_name)
            if self.use_head:
                self.head = ContrastiveHead(getattr(self.lm.config, 'd_model', self.lm.config.hidden_size), config.fpd.output_dim)
        dist_logger.addHandler(logging.FileHandler(os.path.join(config.output_dir, 'dist_log.txt')))

    def create_optimizer(self):
        param_groups = []
        if not self.config.fpd.freeze_lm:
            param_groups.append({'params': self.lm.parameters()})
        if self.head is not None:
            param_groups.append({'params': self.head.parameters(), 'lr': self.config.fpd.lr * self.config.fpd.lr_scale})

        optimizer = torch.optim.Adam(
            param_groups,
            lr=self.config.fpd.lr
        )
        return optimizer

    def forward(self, mode, *inputs):
        # for flop analysis only
        if mode == 'get_reps':
            return self.get_reps(*inputs)
        elif mode == 'infer_pred_forget_with_reps_logit_single':
            return self.infer_pred_forget_with_reps_logit_single(*inputs)
        elif mode == 'pred_forget_with_reps':
            return self.pred_forget_with_reps(*inputs)
        elif mode == 'infer_pred_forget_with_reps_logit_single':
            return self.infer_pred_forget_with_reps_logit_single_profile(*inputs)
        else:
            raise ValueError(mode)

    def get_reps(self, input_ids, attention_mask, labels, decoder_attention_mask, all_ts=False):

        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                          output_hidden_states=True)
        decoder_input_len = decoder_attention_mask.sum(-1) # [B]
        if all_ts:
            raw_rep = outputs.decoder_hidden_states[-1] # [B,T,H]
        else:
            last_layer_hidden = outputs.decoder_hidden_states[-1] # [B,T,H]
            raw_rep = last_layer_hidden[torch.arange(last_layer_hidden.size(0)),decoder_input_len - 1,:]

        if self.config.fpd.freeze_lm:
            raw_rep = raw_rep.detach()

        if self.use_head:
            rep = self.head(raw_rep)
        else:
            rep = raw_rep

        if self.normalize:
            rep = F.normalize(rep, dim=-1)

        return rep

    def get_rep_prod(self, rep_a, rep_b):
        #print(rep_a, rep_b)
        # [B,H], [B,H]
        if self.config.fpd.use_cos_dist:
            rep_prod = F.cosine_similarity(rep_a, rep_b, dim=-1) # [B,T1*T2]
        else:
            if self.config.fpd.sum_or_mean == 'sum':
                rep_prod = (rep_a * rep_b).sum(-1)
            else:
                rep_prod = (rep_a * rep_b).mean(-1)
        rep_prod = - rep_prod / self.temp
        return rep_prod

    def get_rep_prod_mat(self, all_ocl_reps, all_pt_reps):
        # [N2, H], [N1,H]
        if self.config.fpd.use_cos_dist:
            assert all_ocl_reps.size(0) == 1
            rep_prod_grid = F.cosine_similarity(all_ocl_reps, all_pt_reps) # [NT1]
            rep_prod_grid = rep_prod_grid.unsqueeze(0)
        else:
            if self.config.fpd.sum_or_mean == 'sum':
                rep_prod_grid = torch.matmul(all_ocl_reps, all_pt_reps.transpose(0,1))
            else:
                rep_prod_grid = torch.matmul(all_ocl_reps, all_pt_reps.transpose(0,1)) / float(all_ocl_reps.size(1))
        rep_prod_grid = - rep_prod_grid / self.temp
        return rep_prod_grid


    def pred_forget_pairwise(self, input_ids_pt, input_ids_ocl, attention_mask_pt, attention_mask_ocl, labels_pt, labels_ocl,
                decoder_attention_mask_pt, decoder_attention_mask_ocl, priors=None, forget_label=None, **kwargs):
        rep_a = self.get_reps(input_ids_pt, attention_mask_pt, labels_pt, decoder_attention_mask_pt)
        rep_b = self.get_reps(input_ids_ocl, attention_mask_ocl, labels_ocl, decoder_attention_mask_ocl)
        logit = self.get_rep_prod(rep_a, rep_b)

        logit = self.add_bias_to_logit_if_needed(logit, priors)

        prob = F.sigmoid(logit)
        loss = None
        #print(forget_label, prob)

        weights = torch.where(forget_label == 1, self.config.fpd.ce_loss_pos_weight, 1.)

        if forget_label is not None:
            loss = F.binary_cross_entropy(prob, forget_label.float(), weight=weights)
        return prob, loss

    def pred_forget_with_reps(self, all_ocl_reps, all_pt_reps, all_priors, thres=0.5):
        logits = self.get_rep_prod_mat(all_ocl_reps, all_pt_reps)
        logits = self.add_bias_to_logit_if_needed(logits, all_priors)

        prob_grid = F.sigmoid(logits) # [N2,N1]

        preds = (prob_grid > thres).long()
        #print(prob_grid, preds)
        return prob_grid, preds

    def mask_pad_in_labels(self, labels):
        ret = labels.masked_fill(labels==self.tokenizer.pad_token_id, -100)
        return ret

    def clean_batch(self, batch):
        batch['input_ids_pt'], batch['attention_mask_pt'] = trim_batch(batch['input_ids_pt'],
                                                                 self.tokenizer.pad_token_id, batch['attention_mask_pt'])
        batch['input_ids_ocl'], batch['attention_mask_ocl'] = trim_batch(batch['input_ids_ocl'],
                                                                 self.tokenizer.pad_token_id, batch['attention_mask_ocl'])
        batch['labels_pt'], batch['decoder_attention_mask_pt'] = trim_batch(batch['labels_pt'],
                                                                 self.tokenizer.pad_token_id, batch['decoder_attention_mask_pt'])
        batch['labels_ocl'], batch['decoder_attention_mask_ocl'] = trim_batch(batch['labels_ocl'],
                                                                   self.tokenizer.pad_token_id, batch['decoder_attention_mask_ocl'])

        batch['labels_ocl'] = self.mask_pad_in_labels(batch['labels_ocl'])
        batch['labels_pt'] = self.mask_pad_in_labels(batch['labels_pt'])

        batch.pop('input_ids')
        batch.pop('attention_mask')

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda()

        return batch

    def clean_batch_for_rep(self, batch):
        #batch['labels'] = self.mask_pad_in_labels(batch['labels'])
        batch['input_ids'], batch['attention_mask'] = trim_batch(batch['input_ids'], self.tokenizer.pad_token_id, batch['attention_mask'])
        batch['labels'], batch['decoder_attention_mask'] = trim_batch(batch['labels'], self.tokenizer.pad_token_id, batch['_decoder_attention_mask'])
        batch['labels'] = self.mask_pad_in_labels(batch['labels'])
        batch = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']}

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda()

        return batch

    def convert_priors_to_bias(self, priors):
        if self.config.fpd.prior == 'odd':
            bias = torch.log(priors)
        else:
            raise NotImplementedError
        return bias

    def add_bias_to_logit_if_needed(self, logit, priors):
        if self.config.fpd.prior == 'odd':
            bias = self.convert_priors_to_bias(priors)
            logit = logit + bias
        return logit

    def get_rep_dists_mask(self, attn_mask_a, attn_mask_b):
        # [B,T1], [B,T2] -> res[b, T1, T2]
        mask = torch.ones(attn_mask_a.size(0), attn_mask_a.size(1), attn_mask_b.size(1)).to(attn_mask_a.device) # [B,T1,T2]
        for b in range(attn_mask_a.size(0)):
            mask[b, ~(attn_mask_a[b].bool())] = 0
            mask[b, :, ~(attn_mask_b[b].bool())] = 0
        return mask

    def get_rep_dists_mask_1vn(self, pt_dec_attn_mask, ocl_dec_attn_mask):
        #  [N,T1], [1,T2] -> res[N,T1,T2]
        assert ocl_dec_attn_mask.size(0) == 1
        mask = torch.ones(pt_dec_attn_mask.size(0), pt_dec_attn_mask.size(1), ocl_dec_attn_mask.size(1)).to(pt_dec_attn_mask.device)
        mask[~(pt_dec_attn_mask.bool())] = 0
        mask[:,:,~(ocl_dec_attn_mask[0].bool())] = 0
        return mask

class ForgetPredictionModelForCausualLM(ForgettingPredictionModel):
    def __init__(self, config, tokenizer, helper, init_model=True):
        super().__init__(config, tokenizer, helper, init_model=False)
        self.is_sent_encoder = 'MiniLM' in config.fpd.model_name
        if init_model:
            if self.is_sent_encoder:
                self.lm = AutoModel.from_pretrained(config.fpd.model_name, trust_remote_code=True)
            else:
                self.lm = AutoModelForCausalLM.from_pretrained(config.fpd.model_name, trust_remote_code=True)
            if self.use_head:
                self.head = ContrastiveHead(getattr(self.lm.config, 'd_model', self.lm.config.hidden_size), config.fpd.output_dim)

    def get_reps(self, input_ids, all_ts=False, **kwargs):
        input_ids = input_ids.cuda()
        if self.is_sent_encoder:
            outputs = self.lm(input_ids=input_ids)
            hidden = outputs[0]
        else:
            outputs = self.lm(input_ids=input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
        input_len = (input_ids!= self.tokenizer.pad_token_id).sum(-1)
        if all_ts:
            raw_rep = hidden # [B,T,H]
        else:
            raw_rep = hidden[torch.arange(hidden.size(0)), input_len - 1,:]

        if self.config.fpd.freeze_lm:
            raw_rep = raw_rep.detach()

        if self.use_head:
            rep = self.head(raw_rep)
        else:
            rep = raw_rep

        if self.normalize:
            rep = F.normalize(rep, dim=-1)

        return rep

    def pred_forget_pairwise_mse(self, input_ids_pt, input_ids_ocl, forget_label=None, **kwargs):
        rep_a = self.get_reps(input_ids_pt)
        rep_b = self.get_reps(input_ids_ocl)
        score = self.get_rep_prod(rep_a, rep_b)
        #print(forget_label, prob)

        loss = None
        if forget_label is not None:
            loss = F.mse_loss(score, forget_label.float())
        return score, loss

    def pred_forget_pairwise_ce(self, input_ids_pt, input_ids_ocl, forget_label=None, **kwargs):
        rep_a = self.get_reps(input_ids_pt)
        rep_b = self.get_reps(input_ids_ocl)
        logit = self.get_rep_prod(rep_a, rep_b)

        prob = F.sigmoid(logit)
        loss = None
        weights = torch.where(forget_label == 1, self.config.fpd.ce_loss_pos_weight, 1.)

        if forget_label is not None:
            loss = F.binary_cross_entropy(prob, forget_label.float(), weight=weights)
        return prob, loss

    def pred_forget_with_reps_score(self, all_ocl_reps, all_pt_reps, thres=0.):
        scores = self.get_rep_prod_mat(all_ocl_reps, all_pt_reps)
        preds = (scores > thres).long()
        #print(prob_grid, preds)
        return scores, preds

    def batch_to_cuda(self, batch):
        batch_ = {k:v for k,v in batch.items()}
        for k,v in batch_.items():
            if torch.is_tensor(v):
                batch_[k] = v.cuda()
        return batch_