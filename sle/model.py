import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from scipy.stats import entropy
from sklearn.metrics import mean_absolute_error
from transformers import AdamW, RobertaTokenizer, RobertaForSequenceClassification


# TODO: Inference function


class RobertaFinetuner(pl.LightningModule):

    def __init__(self, model_name_or_path='roberta-base', tokenizer=None, params=None):
        super().__init__()

        # saves params to the checkpoint and in self.hparams
        self.save_hyperparameters(params)

        num_labels = 1 # regression
        self.hparams["num_labels"] = num_labels

        self.model = RobertaForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
        print(f"Initial RobertaForSequenceClassification model loaded from {model_name_or_path}.")

        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = tokenizer

        # training loss cache to log mean every n steps
        self.train_losses = []

        if "hidden_dropout_prob" in self.hparams and self.hparams.hidden_dropout_prob is not None:
            self.model.config.hidden_dropout_prob = self.hparams.hidden_dropout_prob

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)
    
    def training_step(self, batch, batch_idx):
        output = self.model(**{k:v for k, v in batch.items() if k not in ["doc_ids"]}, return_dict=True)
        loss = output["loss"]
        self.train_losses.append(loss)

        # logging mean loss every `n` steps
        if batch_idx % int(self.hparams.train_check_interval * self.trainer.num_training_batches) == 0:
            avg_loss = torch.stack(self.train_losses).mean()
            self.log("train_loss", avg_loss)
            self.train_losses = []

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self.model(**{k:v for k, v in batch.items() if k not in ["doc_ids"]}, return_dict=True)
        loss = output["loss"]
        logits = output["logits"].cpu()

        output = {
            "loss": loss,
            "preds": logits,
        }

        output["labels"] = batch["labels"]
        if self.has_param("log_doc_mae"):
            output["doc_ids"] = batch["doc_ids"]

        return output

    def validation_epoch_end(self, outputs, prefix="val"):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(f"{prefix}_loss", loss)

        flat_preds = [y.item() for ys in outputs for y in ys["preds"]]
        flat_labels = [y.item() for ys in outputs for y in ys["labels"]]

        # log entropies
        density, _ = np.histogram(flat_preds, density=True, bins=20)
        self.log(f"{prefix}_entropy", entropy(density, base=2))
        class_preds = [[] for _ in range(5)] # assuming 5 classes (0-4)
        for i in range(len(flat_preds)):
            class_preds[int(flat_labels[i])].append(flat_preds[i])
        ents = []
        for l in range(self.model.num_labels):
            density, _ = np.histogram(class_preds[l], density=True, bins=20)
            ents += [entropy(density, base=2)]
        self.log(f"{prefix}_macro_entropy", np.mean(ents))

        # compute document-level MAE when doing regression
        if self.has_param("log_doc_mae"):
            doc_ids = [y for ys in outputs for y in ys["doc_ids"]]
            doc_preds = {}
            doc_labs = {}
            for i in range(len(doc_ids)):
                if doc_ids[i] not in doc_preds:
                    doc_preds[doc_ids[i]] = [flat_preds[i]]
                    doc_labs[doc_ids[i]] = [flat_labels[i]]
                else:
                    doc_preds[doc_ids[i]].append(flat_preds[i])
                    doc_labs[doc_ids[i]].append(flat_labels[i])
            doc_means = []
            doc_gts = []
            for k, v in doc_preds.items():
                doc_means.append(np.mean(v))
                doc_gts.append(np.mean(doc_labs[k]))
            self.log(f"{prefix}_doc_mae", mean_absolute_error(doc_gts, doc_means))

        return {f"{prefix}_loss": loss}
    
    def validation_epoch_end(self, outputs, prefix="val"):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(f"{prefix}_loss", loss)
        
        flat_preds = [y.item() for ys in outputs for y in ys["preds"]]
        flat_labels = [y.item() for ys in outputs for y in ys["labels"]]

        # log entropies
        density, _ = np.histogram(flat_preds, density=True, bins=20)
        self.log(f"{prefix}_entropy", entropy(density, base=2))
        class_preds = [[] for _ in range(5)] # assuming 5 classes (0-4)
        for i in range(len(flat_preds)):
            class_preds[int(flat_labels[i])].append(flat_preds[i])
        ents = []
        for l in range(self.model.num_labels):
            density, _ = np.histogram(class_preds[l], density=True, bins=20)
            ents += [entropy(density, base=2)]
        self.log(f"{prefix}_macro_entropy", np.mean(ents))

        # compute document-level MAE when doing regression
        if self.has_param("log_doc_mae"):
            doc_ids = [y for ys in outputs for y in ys["doc_ids"]]
            doc_preds = {}
            doc_labs = {}
            for i in range(len(doc_ids)):
                if doc_ids[i] not in doc_preds:
                    doc_preds[doc_ids[i]] = [flat_preds[i]]
                    doc_labs[doc_ids[i]] = flat_labels[i]
                else:
                    doc_preds[doc_ids[i]].append(flat_preds[i])
            doc_means = []
            doc_gts = []
            for k, v in doc_preds.items():
                doc_means.append(np.mean(v))
                doc_gts.append(doc_labs[k])
            self.log(f"{prefix}_doc_mae", mean_absolute_error(doc_gts, doc_means))

        return {f"{prefix}_loss": loss}

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.01
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        
        # use a learning rate scheduler if specified
        if self.hparams.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }

        return optimizer

    def has_param(self, param):
        """Check if param exists and has a non-negative/null value."""
        if param in self.hparams:
            param = self.hparams[param] # set `param` to actual value
            if param is not None:
                if not isinstance(param, bool) or param:
                    return True
        return False

    def save_model(self, path):
        # add inference parameters to model config
        self.model.config.update({p: self.hparams[p] for p in INF_PARAMS})

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"{type(self.model)} model saved to {path}.")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument("--name", type=str, default=None, required=False,)
        parser.add_argument("--project", type=str, default=None, required=False,)
        parser.add_argument("--save_dir", type=str, default=None, required=False,)
        parser.add_argument("--checkpoint", type=str, default=None, required=False,)
        parser.add_argument("--wandb_id", type=str, default=None, required=False,)
        
        parser.add_argument("--train_file", type=str, default=None, required=False)
        parser.add_argument("--val_file", type=str, default=None, required=False)
        parser.add_argument("--x_col", type=str, default="x", required=False,)
        parser.add_argument("--y_col", type=str, default="y", required=False,)
        parser.add_argument("--train_check_interval", type=float, default=0.20)
        parser.add_argument("--train_split", type=float, default=0.9)
        parser.add_argument("--val_split", type=float, default=0.05)

        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--max_samples", type=int, default=-1)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--train_workers", type=int, default=8)
        parser.add_argument("--val_workers", type=int, default=8)
        parser.add_argument("--max_length", type=int, default=128)
        parser.add_argument("--lr_scheduler", action="store_true")
        parser.add_argument("--ckpt_metric", type=str, default="val_loss", required=False,)

        parser.add_argument("--hidden_dropout_prob", type=float, default=None, required=False,)

        parser.add_argument("--log_doc_mae", action="store_true")

        return parser
