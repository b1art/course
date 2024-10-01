import torch
from transformers import BertForSequenceClassification

def load_model(path='bert_model.pth', model_name='bert-base-uncased', num_labels=3):
    """
    Загружает сохранённую модель.
    
    :param path: Путь к сохранённой модели (по умолчанию 'bert_model.pth')
    :param model_name: Название предобученной модели (по умолчанию 'bert-base-uncased')
    :param num_labels: Количество классов для классификации
    :return: Загруженная модель
    """
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f'Модель успешно загружена из файла: {path}')
    return model
