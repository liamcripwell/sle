# Simplicity Level Estimate (SLE) Metric

This repo contains code for the SLE metric as described in the [2023 EMNLP paper](https://aclanthology.org/2023.emnlp-main.739/).

```bash
pip install -e .
```

## Usage
SLE scores can be calculated within python as shown in the example below.

For a raw estimation of a sentence's simplicity, use `'sle'`, but to evaluate sentence simplification systems we recommend providing the input sentences and using `'sle_delta'` ($\Delta \text{SLE}$). See the paper for further details.

A pretrained version of the model described in the paper is available on [HuggingFace](https://huggingface.co/liamcripwell/sle-base).

```python
from sle.scorer import SLEScorer

scorer = SLEScorer("liamcripwell/sle-base")

texts = [
  "Here is a simple sentence.",
  "Here is an additional sentence that makes use of more complex terminology."
]

# raw simplicity estimates
results = scorer.score(texts)
print(results) # {'sle': [3.9842946529388428, 0.5840105414390564]}

# delta from input sentences
results = scorer.score([texts[0]], inputs=[texts[1]])
print(results) # {'sle': [3.9842941761016846], 'sle_delta': [3.4002838730812073]}
```

## Label Softening
You can use the label softening feature described in our paper on your own data.

```python
import pandas as pd
from sle.utils import smooth_labels

df = pd.read_csv("some_data.csv") # data with a quantized label column
df_soft = smooth_labels(df, label_col="label", num_labels=5) # assuming 5 original label values
```

## Training Metric
Our training procedure makes use of [PyTorch Lightning](https://lightning.ai/pytorch-lightning) and we primarily use [wandb](https://wandb.ai/site) for logging.

```bash
python sle/scripts/train.py
  --train_file=newselaauto_train.csv
  --val_file=newselaauto_valid.csv
  --x_col=text
  --y_col=reading_level_smooth
  --batch_size=32
  --learning_rate=1e-5
  --val_check_interval=0.25
  --project=sle_project # wandb project
  --name=test_training # wandb run name
  --log_doc_mae # calculates and logs validation doc-level MAE for early stopping as in paper
  --ckpt_metric=val_doc_mae
```

To train a basic model without logging or document MAE you can use the `--no_log` flag.
```bash
python sle/scripts/train.py
  --train_file=newselaauto_train.csv
  --val_file=newselaauto_valid.csv
  --x_col=text
  --y_col=reading_level_smooth
  --batch_size=32
  --learning_rate=1e-5
  --val_check_interval=0.25
  --ckpt_metric=val_loss
  --no_log
  --save_dir=my_ckpts_path/
```

After training a metric model, use `.save_model()` to save in pytorch format to be loaded by the `SLEScorer`.
```python
from sle.model import RobertaFinetuner

model = RobertaFinetuner.load_from_checkpoint("checkpoints/last.ckpt", strict=False)
model.save_model("my_sle_model")
```

## Citation
If you make use of SLE in your project, please cite our paper:

```bibtex
@inproceedings{cripwell-etal-2023-simplicity,
    title = "Simplicity Level Estimate ({SLE}): A Learned Reference-Less Metric for Sentence Simplification",
    author = {Cripwell, Liam  and
      Legrand, Jo{\"e}l  and
      Gardent, Claire},
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.739",
    doi = "10.18653/v1/2023.emnlp-main.739",
    pages = "12053--12059",
    abstract = "Automatic evaluation for sentence simplification remains a challenging problem. Most popular evaluation metrics require multiple high-quality references {--} something not readily available for simplification {--} which makes it difficult to test performance on unseen domains. Furthermore, most existing metrics conflate simplicity with correlated attributes such as fluency or meaning preservation. We propose a new learned evaluation metric {---} SLE {---} which focuses on simplicity, outperforming almost all existing metrics in terms of correlation with human judgements.",
}
```
