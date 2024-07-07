from .lm import SFTDataset, SFTExampleOnlyDataset

def load_ocl_ds_by_task_id(config, tokenizer, task_cat, task_id):
    if task_cat == 'mmlu':
        all_tasks = config.mmlu_tasks
        task = all_tasks[task_id]
        train_ds = SFTDataset.from_mmlu([task], 'val', config, tokenizer)
        test_ds = SFTDataset.from_mmlu([task], 'test', config, tokenizer)
    elif task_cat == 'bbh':
        all_tasks = config.bbh_tasks
        task = all_tasks[task_id]
        train_ds = SFTDataset.from_bbh([task], 'train', config, tokenizer)
        test_ds = SFTDataset.from_bbh([task], 'eval', config, tokenizer)
    elif task_cat == 'flan':
        all_tasks = config.flan_tasks
        task = all_tasks[task_id]
        train_ds = SFTDataset.from_flan_by_task([task],'train',config, tokenizer)
        test_ds = SFTDataset.from_flan_by_task([task],'validation',config,tokenizer)
    elif task_cat == 'truthful_qa':
        all_tasks = config.truthful_qa_tasks
        task = all_tasks[task_id]
        train_ds = SFTDataset.from_truthful_qa([task],'validation',config, tokenizer)
        test_ds = SFTDataset.from_truthful_qa([task],'test',config,tokenizer) 
    elif task_cat == 'tulu_train':
        all_tasks = config.tulu_tasks
        task = all_tasks[task_id]
        train_ds = SFTDataset.from_tulu_train([task], 'train', config, tokenizer)
        test_ds = SFTDataset.from_tulu_train([task], 'dev', config, tokenizer)  
    else:
        raise NotImplementedError
    return train_ds, test_ds, task

def load_ocl_example_only_ds_by_task_id(config, tokenizer, task_cat, task_id, include_gt):
    if task_cat == 'mmlu':
        all_tasks = config.mmlu_tasks
        task = all_tasks[task_id]
        train_ds = SFTExampleOnlyDataset.from_mmlu([task], 'val', config, tokenizer, include_gt=include_gt)
        test_ds = SFTExampleOnlyDataset.from_mmlu([task], 'test', config, tokenizer, include_gt=include_gt)
    else:
        raise NotImplementedError
    return train_ds, test_ds, task

