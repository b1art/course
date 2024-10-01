import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer

class BERTTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.texts = dataframe['sent'].values
        self.labels = dataframe['level'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_bert_data(file_path, max_len=128):
    dataframe = pd.read_csv(file_path, sep=';')

    # Балансировка датасета (опционально, если необходимо)
    X = dataframe['sent'].values
    y = dataframe['level'].values
    # здесь можно добавить код для балансировки, если нужно

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = BERTTextDataset(pd.DataFrame({'sent': X, 'level': y}), tokenizer, max_len=max_len)

    return dataset, tokenizer
