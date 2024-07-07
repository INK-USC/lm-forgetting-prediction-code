import json
import os
import csv

TASK_DESC = "The following are multiple choice questions (with answers) about {TASK}."

class MMLUHelper:
    def __init__(self, config, task):
        self.config = config
        self.path = config.mmlu.path
        self.cot_prompt_path = config.mmlu.cot_prompt_path
        self.task = task
        self.few_shot_k = config.mmlu.few_shot_k

        self.fs_training_examples = self.load_few_shot_training_examples()

    def row2example(self, row):
        return {
            'question': row[0],
            'choices': row[1:5],
            'gt': row[5]
        }

    def load_few_shot_training_examples(self):
        with open(os.path.join(self.path, 'dev', f'{self.task}_dev.csv')) as f:
            rows = [_ for _ in csv.reader(f)]
        examples = [self.row2example(row) for row in rows]
        return examples

    def get_prompt(self, task, cot, answer_type, is_few_shot):
        if not is_few_shot:
            return self.get_zero_shot_prompt_no_cot(task)
        else:
            if cot:
                return self.get_few_shot_cot_prompt(task)
            else:
                return self.get_few_shot_prompt_no_cot(task, answer_type)

    def get_few_shot_prompt_no_cot(self, task, answer_type='choice'):
        assert answer_type in ['choice','text']
        assert self.few_shot_k <= len(self.fs_training_examples)
        train_set = self.fs_training_examples[:self.few_shot_k]

        prompt = TASK_DESC.format(TASK=task) + ' '
        for example in train_set:
            choices = example['choices']
            prompt += '\n\n'
            prompt += 'Q: ' + example['question'] + '\n'

            if answer_type == 'choice':
                prompt += '(A) {} (B) {} (C) {} (D) {}'.format(choices[0], choices[1], choices[2], choices[3]) + '\n'
                prompt += 'A: ({})'.format(example['gt'])
            else:
                prompt += 'OPTIONS:\n - {}\n- {}\n- {}\n- {}'.format(choices[0], choices[1], choices[2], choices[3]) + '\n'
                answer = choices[ord(example['gt']) - ord('A')]
                prompt += 'A: {}'.format(answer)
        return prompt

    def get_few_shot_cot_prompt(self, task):
        with open(self.cot_prompt_path) as f:
            data = json.load(f)
        return data[task]

    def get_zero_shot_prompt_no_cot(self, task):
        prompt = TASK_DESC.format(TASK=task) + ' '
        return prompt

    def create_examples(self, split, prompt, cot=False, answer_type='choice', example_format='p3'):
        assert split in ['dev','val','test']
        assert answer_type in ['choice','text']
        with open(os.path.join(self.path, split, '{}_{}.csv'.format(self.task, split))) as f:
            rows = [_ for _ in csv.reader(f)]
        raw_examples = [self.row2example(row) for row in rows]
        examples = []
        for raw_example in raw_examples:
            input_text = prompt + '\n\n' + 'Q: ' + raw_example['question'] + '\n'
            choices = raw_example['choices']

            if answer_type == 'choice':
                input_text += '(A) {} (B) {} (C) {} (D) {}'.format(choices[0], choices[1], choices[2], choices[3]) + '\n'
            else:
                input_text += 'OPTIONS:\n - {}\n- {}\n- {}\n- {}'.format(choices[0], choices[1], choices[2],
                                                                   choices[3]) + '\n'

            if cot:
                input_text += "A: Let's think step by step. "
            else:
                input_text += 'A: '

            if answer_type == 'choice':
                answer = '({})'.format(raw_example['gt'])
            else:
                answer = choices[ord(raw_example['gt']) - ord('A')]
            if example_format == 'p3':
                examples.append([input_text, [answer], self.task]) # format of p3
            else:
                examples.append([input_text, answer, self.task])
        return examples
