import numpy as np
import torch

class EditBatchSampler:
    def __init__(self, n, n_edits=1, memorize_mode=False, loc_disjoint=True, seed=0):
        self.memorize_mode = memorize_mode
        self.n = n
        self.n_edits = n_edits
        self.loc_disjoint = loc_disjoint
        self.rng = np.random.default_rng(seed)
        self._init()

    def _init(self):
        self.perm = self.rng.permutation(self.n)
        self.edit_position = 0

    def sample(self, batch_size):
        assert (
            batch_size > self.n_edits
        ), "Batch size is interpreted such that batch_size = n_edits + n_loc"

        if self.memorize_mode:
            return list(range(self.n_edits)), list(range(batch_size - self.n_edits))

        if self.edit_position >= self.n:
            self._init()

        edit_idxs = self.perm[self.edit_position: self.edit_position + self.n_edits]
        self.edit_position += self.n_edits

        loc_idxs = self.rng.choice(self.n, batch_size - self.n_edits)
        if self.loc_disjoint:
            while len(np.intersect1d(edit_idxs, loc_idxs)) > 0:
                loc_idxs = self.rng.choice(self.n, batch_size - self.n_edits)

        return edit_idxs.tolist(), loc_idxs.tolist()

def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v

    return new_dict

def truncate_prefix(tokenizer, inputs, max_len):
    encoding = tokenizer.batch_encode_plus(inputs)
    for i in range(len(encoding.input_ids)):
        length = len(encoding.input_ids[i])
        if length < max_len:
            encoding.input_ids[i] += [tokenizer.pad_token_id] * (max_len - length)
            encoding.attention_mask[i] += [0] * (max_len - length)
        else:
            encoding.input_ids[i] = encoding.input_ids[i][-max_len:]
            encoding.attention_mask[i] = encoding.attention_mask[i][-max_len:]
    return encoding

def apply_chat_template(examples, tokenizer):
    chats = [[
        {'role': 'user', 'content': example[0]},
        {'role': 'assistant', 'content': example[1]}
    ] for example in examples]

    input_texts = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in chats]
    return input_texts

def apply_chat_template_for_generation(examples, tokenizer):
    chats = [[
        {'role': 'user', 'content': example[0]},
    ] for example in examples]

    input_texts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chats]
    return input_texts
