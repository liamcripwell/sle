# Simplicity Level Estimate (SLE) Metric

```bash
pip install -e .
```

## Usage
SLE scores can be calculated within python as shown in the example below.

For a raw estimation of a sentence's simplicity, use `'sle'`, but to evaluate sentence simplification systems we recommend providing the input sentences and using `'sle_delta'` ($\Delta \text{SLE}$). See the paper for further details.

```python
from sle.scorer import SLEScorer

scorer = SLEScorer("path_to_model/")

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
