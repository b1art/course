import torch

def save_model(model, path='bert_model.pth'):
    """
    Сохраняет обученную модель.
    
    :param model: Модель для сохранения
    :param path: Путь для сохранения модели (по умолчанию 'bert_model.pth')
    """
    torch.save(model.state_dict(), path)
    print(f'Модель успешно сохранена по пути: {path}')
