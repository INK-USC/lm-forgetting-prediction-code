from transformers.trainer import Trainer
from .memory_utils import DatasetMemory
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from .utils import trim_batch
from tqdm import tqdm
import logging
from torch.nn import functional as F
import pickle
import numpy as np

replay_logger = logging.getLogger('main')

def score_minus(a: tuple, b: tuple):
    ret = tuple([x - y for x, y in zip(a,b)])
    return ret

class MyTrainer(Trainer):
    def __init__(self, **kwargs):
        self.config = kwargs.pop('config')
        self.memory = kwargs.pop('memory')
        self.past_model_creator = kwargs.pop('past_model_creator', None)
        if self.config.cl_use_distill:
            self.past_model = self.past_model_creator(self.config).cuda()
        else:
            self.past_model = None

        if self.config.use_coreset:
            self.coreset_info = self.load_coreset_info()
        else:
            self.coreset_info = None

        if self.config.use_pred_forget_mat:
            self.taskid2forget = self.load_task2forget()
        else:
            self.taskid2forget = None

        super().__init__(**kwargs)

    def load_coreset_info(self):
        with open(self.config.coreset_bin_dir,'rb') as f:
            coreset_info = pickle.load(f)
        return coreset_info

    def load_task2forget(self, split='test'):
        with open(self.config.pred_forget_file,'rb') as f:
            obj = pickle.load(f)

        task_id2forget = {}
        for idx, task_id in enumerate(obj[f'{split}_ocl_idxs']):
            task_id2forget[task_id] = obj[f'{split}_mat'][idx]
        return task_id2forget

    def remove_keys(self, batch):
        return {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels', 'label_ids']}

    def mask_pad_in_labels(self, labels):
        ret = labels.masked_fill(labels==self.tokenizer.pad_token_id, -100)
        return ret

    def clean_batch(self, batch, phase='unknown'):
        batch = {k: v for k,v in batch.items()}
        # if not self.config.is_seq2seq:
        #     if phase == 'train':
        #         batch['input_ids'] = batch['labels']
        #         batch['attention_mask'] = batch['_decoder_attention_mask']
        batch['labels'] = self.mask_pad_in_labels(batch['labels'])

        if self.config.is_seq2seq:
            batch['input_ids'], batch['attention_mask'] = trim_batch(batch['input_ids'],
                                                                     self.tokenizer.pad_token_id, batch['attention_mask'])

        batch_ = self.remove_keys(batch)
        return batch_

    def batch_to_gpu(self, batch):
        batch['labels'] = self.mask_pad_in_labels(batch['labels'])
        batch['input_ids'], batch['attention_mask'] = trim_batch(batch['input_ids'],
                                                                 self.tokenizer.pad_token_id, batch['attention_mask'])
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda()
        return batch

    def ocl_train_single_task_er(self, train_loader, optimizer=None, on_epoch_end=None):
        config = self.config
        model = self.model
        num_epoch = config.num_epoch_per_task
        max_step = config.max_step_per_task

        total_step = min(num_epoch * len(train_loader), max_step)
        if max_step == -1:
            total_step = num_epoch * len(train_loader)

        total_epoch = (total_step - 1) // len(train_loader) + 1

        do_replay = config.do_replay
        do_candidate = config.do_candidate
        candidate = None

        replay_freq = config.replay_freq
        replay_k = config.replay_k

        if total_step < 0:
            raise ValueError(num_epoch, max_step, total_step)

        # must reset scheduler
        scheduler = self.create_scheduler(total_step, optimizer)
        global_step = 0

        model.zero_grad()
        train_end = False

        print('Total step {}, total epoch {}'.format(total_step, total_epoch))

        for epoch in range(total_epoch):
            for _, batch in tqdm(enumerate(train_loader), desc="Epoch {}".format(epoch)):
                global_step += 1
                if global_step > total_step:
                    train_end = True
                    break

                #batch['labels'] = self.mask_pad_in_labels(batch['labels'])
                #batch_ = self.remove_keys(batch)
                batch_ = self.clean_batch(batch)
                loss_dt = self.training_step(model, batch_)

                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.max_grad_norm,
                    )

                optimizer.step()
                scheduler.step()
                model.zero_grad()

                # replay if needed
                if do_replay and global_step % replay_freq == 0:
                    if do_candidate and candidate is not None:
                        mem_batch = self.choose_from_candidates(candidate, k=replay_k)
                        candidate = None
                    else:
                        mem_batch, sampled_tasks, sampled_idxs = self.memory.random_sample(k=replay_k)

                    mem_batch_ = self.clean_batch(mem_batch)
                    mem_loss_dt = self.training_step(model, mem_batch_)

                    if config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config.max_grad_norm,
                        )

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                # check if candidate is None
                if do_candidate and candidate is None:
                    candidate = self.prepare_replay_candidates()
            if on_epoch_end is not None:
                on_epoch_end()
            if train_end:
                break

    def get_raw_gradients(self, model, batch):
        all_grads = {}
        model.zero_grad()
        batch_ = self.clean_batch(batch)
        loss = self.training_step(model, batch_)
        for n, p in model.named_parameters():
            if p.grad is not None:
                all_grads[n] = p.cpu().detach()
        return all_grads

    def model_update_over_stream(self, model, ocl_batches, optimizer, n_step, eval_mode=False):
        config = self.config
        for batch in ocl_batches:
            batch_ = self.clean_batch(batch)
            if eval_mode:
                loss_dt = self.training_step_eval_mode(model, batch_)
            else:
                loss_dt = self.training_step(model, batch_)

            #if not config.delay_opt:
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm,
                )

            optimizer.step()
            model.zero_grad()

    def model_update_accum_grad(self, model, batch, step, optimizer,
                                mem_batch=None):
        batch_ = self.clean_batch(batch, phase='train')
        loss_dt = self.training_step(model, batch_)
        loss_mem = None

        if mem_batch is not None:
            mem_batch_ = self.clean_batch(mem_batch, phase='train')
            loss_mem = self.training_step(model, mem_batch_)

        step += 1
        if step % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.max_grad_norm,
                )

            optimizer.step()
            model.zero_grad()
        return loss_dt, step, loss_mem

    def get_replay_batch(self, ds_memory, ocl_task_id):
        config = self.config
        replay_n = config.replay_n if config.replay_n > 0 else config.per_device_train_batch_size
        if self.config.cl_method == 'er':
            batch, tasks, sample_idxs = ds_memory.random_sample(replay_n)
        elif self.config.cl_method == 'mir_pred':
            forget = self.taskid2forget[ocl_task_id]
            forget = np.nan_to_num(forget, nan=-10)
            batch, tasks, sampled_idxs = ds_memory.weight_random_sampling(replay_n, forget, self.config.mir_pred_weight_temp)
        else:
            raise NotImplementedError
        return batch


    def nstep_model_update(self, model, batch, optimizer, n_step, do_replay=False, do_retrieve=False, pt_ds=None,
                           score_func=None, replay_optimizer=None, eval_mode=False, pred_forgets=None, use_pred_forget=False):
        config = self.config
        replay_freq = config.replay_freq
        replay_n = config.replay_n if config.replay_n > 0 else config.per_device_train_batch_size
        replay_n_step = config.replay_n_step
        mir_with_abs_score = config.mir_with_abs_score

        if replay_optimizer is None:
            replay_optimizer = optimizer


        # need this for measuring forgetting independently
        replayed_idxs = []

        if do_replay:

            memory = DatasetMemory(pt_ds, self.data_collator)
            if do_retrieve:
                cand_batch, _, sampled_idxs = memory.random_sample(k=config.cand_k)
                cand_met_scores_before = self.get_scores_no_grad(cand_batch, score_func)
            mem_batch = None

        for step in range(n_step):
            batch_ = self.clean_batch(batch, phase='train')
            if eval_mode:
                loss_dt = self.training_step_eval_mode(model, batch_)
            else:
                loss_dt = self.training_step(model, batch_)


            replay_logger.info('OCL loss: {}'.format(loss_dt.item()))
            should_replay = do_replay and ((replay_freq == -1 and step == n_step - 1) or (replay_freq !=-1 and step % replay_freq == 0))

            if should_replay:
                print('Replay at step {}'.format(step))
                if do_retrieve:
                    if mem_batch is not None and config.mir_no_resample:
                        pass
                    else:
                        cand_met_scores_after = self.get_scores_no_grad(cand_batch, score_func)

                        if mir_with_abs_score:
                            score_drop = cand_met_scores_after # trick
                            score_drop_widx = [(idx, a) for idx, a in enumerate(score_drop)]
                            score_drop_widx.sort(key=lambda x: x[1], reverse=True)
                            top_drop_idxs = [x[0] for x in score_drop_widx[:replay_n]]
                        else:
                            score_drop = [score_minus(before, after) for before, after in zip(cand_met_scores_before, cand_met_scores_after)]
                            score_drop_widx = [(idx, a) for idx, a in enumerate(score_drop)]
                            score_drop_widx.sort(key=lambda x: x[1], reverse=True)
                            top_drop_idxs = [x[0] for x in score_drop_widx[:replay_n]]

                        mem_batch = {}
                        for k, v in cand_batch.items():
                            if type(v) is not list:
                                mem_batch[k] = v[top_drop_idxs]
                            else:
                                mem_batch[k] = [v[idx] for idx in top_drop_idxs]
                        replayed_idxs.extend(top_drop_idxs)

                elif use_pred_forget:
                    mem_batch, _, sampled_idxs = memory.random_sample_from_indices_with_filling(k=replay_n, indices=pred_forgets,
                                                                                                replayed_idxs=replayed_idxs,
                                                                                                no_repeat=config.er_no_repeat)
                    replayed_idxs.extend(sampled_idxs)
                else:
                    if mem_batch is not None and config.mir_no_resample:
                        pass
                    else:
                        mem_batch, _, sampled_idxs = memory.random_sample(k=replay_n)
                        replayed_idxs.extend(sampled_idxs)

                # see whether slicing batches are needed
                mem_bs = len(mem_batch['input_ids'])
                if mem_bs > config.per_device_train_batch_size:
                    mem_batches = self.slice_batch(mem_batch, config.per_device_train_batch_size)
                else:
                    mem_batches = [mem_batch]

                for mem_batch in mem_batches:
                    if config.cl_use_distill:
                        with torch.no_grad():
                            distill_target, _ = self.get_distill_target(self.past_model, mem_batch, eval_mode)
                        distill_target = distill_target.detach()

                    for _ in range(replay_n_step):
                        if config.cl_use_distill:
                            self.distill_replay(config, model, replay_optimizer, mem_batch, distill_target, eval_mode)
                        else:
                            self.simple_replay(config, model, replay_optimizer, mem_batch, eval_mode)

            #if config.delay_opt:
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm,
                )

            optimizer.step()
            model.zero_grad()

        return replayed_idxs

    def simple_replay(self, config, model, replay_optimizer, mem_batch, eval_mode):
        mem_batch_ = self.clean_batch(mem_batch, phase='train')
        if eval_mode:
            loss_mem = self.training_step(model, mem_batch_)
        else:
            loss_mem = self.training_step_eval_mode(model, mem_batch_)

        replay_logger.info('Replay loss: {}'.format(loss_mem.item()))
        return loss_mem

    def kl_soft_loss(self, config, labels, logits, mem_logits):
        mask = (labels != -100).view(-1)
        log_prob = F.log_softmax(logits / config.distill_student_temp, -1)
        mem_log_prob = F.log_softmax(mem_logits / config.distill_teacher_temp, -1)
        log_prob, mem_log_prob = log_prob.view(-1, log_prob.size(2)), mem_log_prob.view(-1, log_prob.size(2))

        kl_div = F.kl_div(log_prob, mem_log_prob, log_target=True, reduction='none')  # [B*T]
        kl_div = kl_div.masked_fill(~ mask.unsqueeze(1), 0.)

        if config.distill_reduction == 'mean':
            mean_kl_div = kl_div.mean(-1).sum() / mask.float().sum()
        elif config.distill_reduction == 'sum':
            mean_kl_div = kl_div.sum() / mask.float().sum()
        else:
            raise ValueError(config.distill_reduction)
        return mean_kl_div

    def distill_replay(self, config, model, replay_optimizer, mem_batch, distill_target, eval_mode):
        new_logits, mem_batch_ = self.get_distill_target(model, mem_batch, eval_mode)
        distill_loss = self.kl_soft_loss(config, mem_batch_['labels'], distill_target, new_logits)
        fin_distill_loss = config.distill_alpha * distill_loss

        fin_distill_loss.backward()
        replay_logger.info('Replay loss: {}'.format(fin_distill_loss.item()))
        return distill_loss

    def get_distill_target(self, model, mem_batch, eval_mode):
        if eval_mode:
            model.eval()
        mem_batch_ = self.clean_batch(mem_batch, phase='train')
        mem_batch_ = {k: v.cuda() for k,v in mem_batch_.items()}
        _, outputs = self.compute_loss(model, mem_batch_, return_outputs=True)
        return outputs.logits, mem_batch_

    def get_before_logits(self, model, batch, eval_mode):
        if eval_mode:
            model.eval()
        batch_ = self.clean_batch(batch)
        batch_ = {k: v.cuda() for k,v in batch_.items()}
        _, outputs = self.compute_loss(model, batch_, return_outputs=True)
        return outputs.logits

    def get_eval_dataloader_raw(self, eval_dataset, batch_size=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size if batch_size is None else batch_size,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def prepare_replay_candidates(self):
        cand_k = self.config.cand_k
        mem_batch, _, sampled_idxs = self.memory.random_sample(k=cand_k)
        losses = self.get_losses_no_grad(mem_batch)
        return mem_batch, losses

    def slice_batch(self, batch, bs):
        batches = []
        total = len(batch['input_ids'])
        for idx in range(0, total, bs):
            new_batch = {}
            for k,v in batch.items():
                new_batch[k] = batch[k][idx:idx+bs]
            batches.append(new_batch)
        return batches

    @torch.no_grad()
    def get_losses_no_grad(self, batch):
        batches = self.slice_batch(batch, bs=self.config.per_device_train_batch_size)
        all_losses = []
        for batch in batches:
            batch = self._prepare_inputs(batch)
            #batch['labels'] = self.mask_pad_in_labels(batch['labels'])
            batch_ = self.clean_batch(batch)
            _, outputs = self.compute_loss(self.model, batch_, return_outputs=True)

            lm_logits = outputs.logits

            loss_fct = CrossEntropyLoss(reduction='none')
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.model.config.vocab_size), batch['labels'].view(-1))
            non_pad_mask = (batch['labels'] != -100).long()
            seq_lens = non_pad_mask.sum(-1)
            lm_losses = masked_lm_loss.view(*batch['labels'].size()).sum(-1)
            lm_losses = lm_losses / (seq_lens + 1e-10)
            all_losses.append(lm_losses)
        all_losses = torch.cat(all_losses, -1)
        return all_losses

    @torch.no_grad()
    def get_scores_no_grad(self, batch, score_func):
        batches = self.slice_batch(batch, bs=self.config.per_device_train_batch_size)
        all_preds, all_gts = [], []
        scores = []
        for batch in batches:
            batch = self._prepare_inputs(batch)
            #batch['labels'] = self.mask_pad_in_labels(batch['labels'])
            batch_ = self.clean_batch(batch)

            outputs = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                num_beams=self.config.num_beams,
                max_length=self.config.max_output_length,
                decoder_start_token_id=self.model.config.decoder_start_token_id
            )

            for b_idx in range(len(batch['input_ids'])):
                pred_ = self.tokenizer.decode(outputs[b_idx])
                pred = self.tokenizer.decode(outputs[b_idx], skip_special_tokens=True)
                gt = batch['original_answers'][b_idx]
                all_preds.append(pred)
                all_gts.append(gt)
                scores.append(score_func(gt, pred))
        return scores


    def choose_from_candidates(self, candidate, k):
        cand_batch, cand_old_losses = candidate

        cand_new_losses = self.get_losses_no_grad(cand_batch)
        loss_increase = cand_new_losses - cand_old_losses
        _, topk_idx = loss_increase.topk(k)

        mem_batch = {}
        for k, v in cand_batch.items():
            if type(v) is not list:
                mem_batch[k] = v[topk_idx]
            else:
                mem_batch[k] = [v[idx] for idx in topk_idx.cpu().numpy().tolist()]

        chosen_tasks = mem_batch['task_name']
        replay_logger.info('Chosen tasks: {}'.format(chosen_tasks))

        return mem_batch

    def training_step_eval_mode(self, model, inputs, **kwargs) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.eval()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False, keep_logit_topk=-1):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        #print(inputs['input_ids'].shape)

        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
