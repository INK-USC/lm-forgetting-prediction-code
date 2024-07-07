from transformers import DataCollatorWithPadding
import re
import numpy as np
import torch

class DataCollatorWithPaddingStr(DataCollatorWithPadding):
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """


    def __call__(self, features):
        features_non_str = []
        features_str = []
        for feature in features:
            dic, dic2 = {}, {}
            for k, v in feature.items():
                if type(v) is not str:
                    dic[k] = v
                else:
                    dic2[k] = v
            features_non_str.append(dic)
            features_str.append(dic2)

        batch = self.tokenizer.pad(
            features_non_str,
            padding='max_length',
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        for dic in features_str:
            for k, v in dic.items():
                if k not in batch:
                    batch[k] = []
                batch[k].append(v)

        return batch

class DataCollatorWithPaddingStrForLM(DataCollatorWithPadding):
    def __call__(self, features):
        features_non_inp = []
        features_inp = []
        for feature in features:
            dic, dic2 = {}, {}
            for k, v in feature.items():
                if k in ['input_ids', 'attention_mask']:
                    dic[k] = v
                else:
                    dic2[k] = v
            features_inp.append(dic)
            features_non_inp.append(dic2)

        batch = self.tokenizer.pad(
            features_inp,
            padding='max_length',
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch['labels'] = batch['input_ids'] # for LM

        for dic in features_non_inp:
            for k, v in dic.items():
                if k not in batch:
                    batch[k] = []
                batch[k].append(v)
        return batch

class DataCollatorMaskedStrForLM(DataCollatorWithPadding):
    def __init__(self, *args, **kwargs):
        self.ans_start_patt = kwargs.pop('ans_start_patt')
        super().__init__(*args, **kwargs)

        self.ans_start_tokens = np.array(self.tokenizer.encode(self.ans_start_patt))

    def find_ans_start(self, input_ids):
        pos = None
        input_ids = np.array(input_ids)
        for i in range(len(input_ids)-1, -1, -1):
            if input_ids[i] == self.ans_start_tokens[0]:
                if (input_ids[i:i+len(self.ans_start_tokens)] == self.ans_start_tokens).all():
                    pos = i
                    break
        return pos

    def make_label(self, input_ids):
        input_ids = np.array(input_ids)
        pos = self.find_ans_start(input_ids)
        labels = np.copy(input_ids)
        if pos is None:
            print('No label found')
            labels[:] = -100
        else:
            labels[:pos + len(self.ans_start_tokens)] = -100
        labels = torch.from_numpy(labels).long()
        return labels

    def __call__(self, features):
        features_non_inp = []
        features_inp = []
        for feature in features:
            dic, dic2 = {}, {}
            for k, v in feature.items():
                if k in ['input_ids', 'attention_mask']:
                    dic[k] = v
                else:
                    dic2[k] = v
            features_inp.append(dic)
            features_non_inp.append(dic2)

        batch = self.tokenizer.pad(
            features_inp,
            padding='max_length',
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        for dic in features_non_inp:
            for k, v in dic.items():
                if k not in batch:
                    batch[k] = []
                batch[k].append(v)

        labels = [self.make_label(input_ids) for input_ids in batch['input_ids']]
        batch['labels'] = torch.stack(labels)


        return batch


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])
