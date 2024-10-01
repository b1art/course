import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Загрузка предобученной модели BERT
class BERTTextErrorClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def tokenize(self, text, max_len=128):
        # Токенизация текста с помощью BERT токенизатора
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return encoding['input_ids'], encoding['attention_mask']

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    def predict(self, text, device='cpu'):
        self.model.eval()
        input_ids, attention_mask = self.tokenize(text)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predicted = torch.max(logits, dim=1)

        return predicted.item()
