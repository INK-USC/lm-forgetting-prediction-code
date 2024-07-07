import json
import os
import csv

TASK_DESC = "The following are multiple choice questions (with answers) about {TASK}."

class BBHHelper:
    def __init__(self, config, task):
        self.config = config
        self.path = config.bbh.path
        self.prompt_path = config.bbh.prompt_path
        self.task = task

    def parse_options(self, text):
        canary = 'Options:\n'
        idx = text.find(canary)
        if idx == -1:
            return None
        option_texts = text[idx + len(canary):]
        options = option_texts.split('\n')
        options = [x.strip() for x in options if x.strip()]

        option2text = {}
        for option in options:
            option2text[option[:3]] = option2text[4:]
        return option2text

    def create_examples(self, split=None, cot=False, example_format='p3'):
        examples = []
        if split is None:
            with open(os.path.join(self.path, '{}.json'.format(self.task))) as f:
                raw_examples = json.load(f)['examples']
        else:
            if split == 'val':
                split = 'eval'
            with open(os.path.join(self.path, '{}_{}.json'.format(self.task, split))) as f:
                raw_examples = json.load(f)['examples']
        with open(os.path.join(self.prompt_path, '{}.txt'.format(self.task))) as f:
            lines = f.readlines()
            #if cot:
            prompt = '\n'.join(lines[2:])
            #else:
            #    prompt = lines[2]

        for raw_example in raw_examples:
            if cot:
                examples.append(
                    [
                        prompt + '\nQ: ' + raw_example['input'] + '\nA: Let\'s think step by step.\n',
                        [raw_example['target']] if example_format == 'p3' else raw_example['target'],
                        self.task
                    ]
                )
            else:
                examples.append(
                    [
                        prompt + '\nQ: ' + raw_example['input'] + '\nA: The short answer is: ',
                        [raw_example['target']] if example_format == 'p3' else raw_example['target'],
                        self.task
                    ]
                )
        return examples