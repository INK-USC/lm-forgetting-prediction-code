from .utils.analysis_tools import initialize
from .utils.config import merge_config_into_args
import argparse
from transformers import TrainingArguments
from .trainer.fgt_prediction_trainer import ForgettingPredictionModel, ForgetPredictionModelForCausualLM
from .data_utils.fpd_helper import FpdP3Helper, DataCollatorWithPaddingStrForFpd
import logging
import os
import torch
from tqdm import tqdm
import numpy as np

logger = logging.getLogger('fpd_main')

def train_fpd_model(config, fpd_model, fpd_optimizer, fpd_helper):
    fpd_model.train()
    fpd_train_step = config.fpd.train_step
    bs = config.fpd.train_batch_size

    best_score = -1
    key_met = 'f1_mean'
    best_state, best_step = {}, -1
    save_step = 1000
    ckpt_step = config.fpd.ckpt_step
    fpd_optimizer.zero_grad()

    for step in range(fpd_train_step):
        if (step + 1) % args.eval_step == 0 or (step == 0 and not args.skip_first_eval):
            val_met, _ = infer_fpd_model_rep_task(config, fpd_model, fpd_helper, 'dev')
            if val_met[key_met] > best_score:
                best_state = {k:v.cpu().clone() for k,v in fpd_model.state_dict().items()}
                best_score = val_met[key_met]
                best_step = step + 1

        if config.fpd.method == 'rep_task_level':
            fpd_batch = fpd_helper.sample_episode_task_level_balanced('train',bs=bs)
            fpd_batch = fpd_model.batch_to_cuda(fpd_batch)
            if config.fpd.binarilize_labels:
                pred_logits, loss = fpd_model.pred_forget_pairwise_ce(**fpd_batch)
            else:
                pred_logits, loss = fpd_model.pred_forget_pairwise_mse(**fpd_batch)
        else:
            raise NotImplementedError

        logger.info(f'Training loss, step {step}, loss {loss.item()}')

        loss = loss / config.fpd.grad_accum
        loss.backward()
        
        if (step + 1) % config.fpd.grad_accum == 0:
            fpd_optimizer.step()
            fpd_optimizer.zero_grad()

        if (step + 1) % save_step == 0:
            torch.save({'step': best_step, 'config': config.to_dict(), 'state': best_state, 'score': best_score},
                       os.path.join(config.output_dir, 'best_model.pt'))

        if ckpt_step > 0 and (step + 1) % ckpt_step == 0:
            torch.save({'step': step, 'config': config.to_dict(), 'state': {k:v.cpu().clone() for k,v in fpd_model.state_dict().items()}},
                       os.path.join(config.output_dir, 'model.{}.pt'.format(step + 1)))


def infer_fpd_model_rep_task(config, fpd_model, fpd_helper: FpdP3Helper, split, save_path=None, try_thres=False):
    print('Starting inference')
    is_training = fpd_model.training
    fpd_model.eval()
    # get reps of all pt and ocl_examples
    pt_loader, ocl_loader = fpd_helper.get_pt_dataloader(split, config.fpd.eval_batch_size), \
                            fpd_helper.get_ocl_dataloader_concat(split, config.fpd.eval_batch_size)
    fgt_label_grid = fpd_helper.get_ground_truth_mat(split)

    # all reps
    all_pt_reps = []
    all_ocl_reps = []

    with torch.no_grad():
        print('Getting PT example reps')
        for idx, pt_batch in tqdm(enumerate(pt_loader), total=len(pt_loader)):
            pt_batch = fpd_model.batch_to_cuda(pt_batch)
            reps = fpd_model.get_reps(pt_batch['input_ids'])
            reps = reps.detach()
            all_pt_reps.append(reps)

        all_pt_reps = torch.cat(all_pt_reps, 0) # [N1,H]
        print('Getting OCL example reps')
        for idx, ocl_batch in tqdm(enumerate(ocl_loader), total=len(ocl_loader)):
            ocl_batch = fpd_model.batch_to_cuda(ocl_batch)
            reps = fpd_model.get_reps(ocl_batch['input_ids'])
            reps = reps.detach()
            all_ocl_reps.append(reps)

        all_ocl_reps = torch.cat(all_ocl_reps, 0) # [N2, H]
        prod_mat = fpd_model.get_rep_prod_mat(all_ocl_reps, all_pt_reps)
        score_grid, bin_preds_grid = fpd_model.pred_forget_with_reps_score(all_ocl_reps, all_pt_reps, thres=0.)
        met_dict = fpd_helper.evaluate_metrics(fgt_label_grid, bin_preds_grid)
    
    logger.info('Metrics over {}: {}'.format(split, met_dict))
    fpd_model.train(is_training)
    if save_path:
        print('Save path is', save_path)
        fpd_helper.save_raw_scores(prod_mat, save_path, split)
    return met_dict, preds_grid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')
    parser.add_argument("--ocl_task")
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--eval_step", default=100, type=int)
    parser.add_argument("--load_model_dir")
    parser.add_argument("--load_model_name", default='best_model.pt')
    parser.add_argument("--skip_first_eval", action='store_true')
    parser.add_argument("--return_pred_logits", action='store_true')
    parser.add_argument("--try_thres", action='store_true')

    args = parser.parse_args()

    config, base_model, tokenizer, base_trainer, collator = initialize(args.config_files, args.templates)
    base_model = base_model.cuda()

    training_args = TrainingArguments(output_dir=config.output_dir)
    merge_config_into_args(config, training_args)


    logger.setLevel(logging.INFO)
    os.makedirs(config.output_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(config.output_dir, 'log.txt')))

    config.save(config.output_dir, 'config.json')

    fpd_collator = DataCollatorWithPaddingStrForFpd(tokenizer)

    fpd_helper = FpdP3Helper(config, tokenizer, fpd_collator, args.ocl_task)

    if config.is_seq2seq:
        fpd_model = ForgettingPredictionModel(config, tokenizer, fpd_helper).cuda()
    else:
        fpd_model = ForgetPredictionModelForCausualLM(config, tokenizer, fpd_helper).cuda()

    fpd_optimizer = fpd_model.create_optimizer()

    if args.load_model_dir:
        load_model_dir = args.load_model_dir
        print('Loading from {}'.format(load_model_dir))

        model_dir = os.path.join(load_model_dir, args.load_model_name)
        save_obj = torch.load(model_dir)
        fpd_model.load_state_dict(save_obj['state'])

    if args.do_train:
        train_fpd_model(config, fpd_model, fpd_optimizer, fpd_helper)

    if args.do_eval:
        if not args.load_model_dir:
            load_model_dir = config.output_dir
        else:
            load_model_dir = args.load_model_dir
        
        print('Loading from {}'.format(load_model_dir))
        model_dir = os.path.join(load_model_dir, args.load_model_name)

        if not os.path.isfile(model_dir):
            logger.info("No trained model found. Evaluating with fresh model.")
        else:
            save_obj = torch.load(model_dir)
            fpd_model.load_state_dict(save_obj['state'])

        met_dict, preds_grid = infer_fpd_model_rep_task(config, fpd_model, fpd_helper, 'dev',
                                                  save_path=os.path.join(config.output_dir, f'fpd_dev/{args.ocl_task}'))
