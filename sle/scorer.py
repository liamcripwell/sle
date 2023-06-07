from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SLEScorer():

    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def score(self, sentences, batch_size=8):
        test_data = DataLoader([[sent] for sent in sentences], batch_size=batch_size, collate_fn=self.collate)

        rls = []
        for batch in tqdm(test_data):
            batch = { k: xi.to(self.device, non_blocking=True) for k, xi in batch.items() }
            output = self.model(**batch, return_dict=True)
            print(output)
            rls += [l.item() for l in output["logits"]]

        return rls

    def collate(self, batch):
        data = self.tokenizer([x[0] for x in batch], max_length=128, padding=True, truncation=True, 
                                add_special_tokens=True, return_tensors='pt')
        
        return data