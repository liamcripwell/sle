import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class SLEDataModule(pl.LightningDataModule):

    def __init__(self, tokenizer, params=None):
        super().__init__()

        self.tokenizer = tokenizer
        self.save_hyperparameters(params)

    def setup(self, stage):
        # read and prepare input data
        self.train = pd.read_csv(self.hparams.train_file).dropna()
        self.train = self.train.sample(frac=1)[:min(self.hparams.max_samples, len(self.train))] # NOTE: this will actually exclude the last item
        if self.has_param("val_file"):
            self.valid = pd.read_csv(self.hparams.val_file).dropna()
        print("All data loaded.")

        # train, validation, test split
        if self.hparams.val_file is None:
            train_span = int(self.hparams.train_split * len(self.train))
            val_span = int((self.hparams.train_split + self.hparams.val_split) * len(self.train))
            self.train, self.valid, self.test = np.split(self.train, [train_span, val_span])
        else:
            self.test = self.train[:16] # arbitrarily have 16 test samples as precaution

        train_seqs = list(self.train[self.hparams.x_col])
        valid_seqs = list(self.valid[self.hparams.x_col])
        test_seqs = list(self.test[self.hparams.x_col])

        if self.has_param("log_doc_mae"):
            # add document ids to allow for high-level logging during regression
            self.train = list(zip(train_seqs, list(self.train["doc_id"]), list(self.train[self.hparams.y_col])))
            self.valid = list(zip(valid_seqs, list(self.valid["doc_id"]), list(self.valid[self.hparams.y_col])))
            self.test = list(zip(test_seqs, list(self.test["doc_id"]), list(self.test[self.hparams.y_col])))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size, shuffle=True, 
                            num_workers=self.hparams.train_workers, pin_memory=True, 
                            collate_fn=self.prepro_collate)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.hparams.batch_size, 
                            num_workers=self.hparams.val_workers, pin_memory=True, 
                            collate_fn=self.prepro_collate)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size, num_workers=1, 
                            pin_memory=False, collate_fn=self.prepro_collate)

    def prepro_collate(self, batch):
        inputs = [x[0] for x in batch]

        # interpret labels if required
        labels = None
        if self.batch_has_labels(batch):
            labels = [x[-1] for x in batch]

        data = self.tokenizer(inputs, max_length=128, padding=True, truncation=True, 
                                add_special_tokens=True, return_tensors='pt')

        if labels is not None:
            data["labels"] = torch.tensor(labels).float()
            if self.has_param("log_doc_mae"):
                data["doc_ids"] = [x[1] for x in batch]

        return data

    def has_param(self, param):
        """Check if param exists and has a non-negative/null value."""
        if param in self.hparams:
            param = self.hparams[param] # set `param` to actual value
            if param is not None:
                if not isinstance(param, bool) or param:
                    return True
        return False

    def get_param(self, param):
        if self.has_param(param):
            return self.hparams[param]

    def batch_has_labels(self, batch):
        # deduce whether batches contain labels
        num_batch_items = len(batch[0])
        num_expected = 1 + sum([self.has_param(p) for p in ["log_doc_mae"]])
        return num_batch_items > num_expected
