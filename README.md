# Predicting Example Forgetting in Language Model Fine-Tuning

This repository contains code implementation for our research on predicting example forgetting in (continual) langauge model fine-tuning.

- [What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement](https://arxiv.org/abs/2402.01865). To appear at ICML 2024 (Spotlight)

- [Demystifying Forgetting in Language Model Fine-Tuning with Statistical Analysis of Example Associations](https://arxiv.org/abs/2406.14026). On Arxiv (June 2024)

☔ Check out our [Project page](https://inklab.usc.edu/lm-forgetting-prediction/) for visualized stastics of forgetting and a brief summary of our approaches and results!

## Installation

We used Python 3.9.12, PyTorch 2.1.2+cu118 and VLLM 0.3.3+cu118. See `requirements.txt` for other requirements.

## Forgetting prediction

The following scripts run forgetting prediction algorithms based on statistics collected in our fine-tuning runs (see `data/stats`)

### Matrix completion approaches (Arxiv)

```
bash scripts/mat_completion/olmo_7b.sh
bash scripts/mat_completion/olmo_7b_inst.sh
```

### Trainable representation dot-product approaches (ICML 2024)
```
bash scripts/rep_dot/train_olmo_7b.sh
bash scripts/rep_dot/train_olmo_7b_inst.sh
```

## Training and collecting statistics of forgetting

See `scripts/run_stat_olmo` and `src/run_stat_olmo_inst` for OLMo-7B and OLMo-7B-Instruct experiments. We first fine-tune models with `train_*` scripts and collect statistics with `*eval_ppl` scripts.


## Replaying examples predicted to be forgotten

We perform experiments with OLMo 7B and Dolma. The following scripts replay random / ground truth forgotten / predicted examples.


```
bash scripts/olmo_ft_replay/replay-random.sh
bash scripts/olmo_ft_replay/replay-gt.sh
bash scripts/olmo_ft_replay/replay-knn.sh
```

## TODO items
- FLAN T5 and BART statistics

## Bibtex
```
@article{Jin2024DemystifyingFI,
  title={Demystifying Forgetting in Language Model Fine-Tuning with Statistical Analysis of Example Associations},
  author={Xisen Jin and Xiang Ren},
  journal={ArXiv},
  year={2024},
  volume={abs/2406.14026},
  url={https://arxiv.org/abs/2406.14026}
}
      
@article{Jin2024WhatWM,
  title={What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement},
  author={Xisen Jin and Xiang Ren},
  journal={ArXiv},
  year={2024},
  volume={abs/2402.01865},
  url={https://arxiv.org/abs/2402.01865}
}

```