import torch
from torch.utils.data import DataLoader, random_split
from bert_model import BERTTextErrorClassifier
from bert_reader import load_bert_data
from bert_learning import train_bert_model, evaluate_bert_model
from save_model import save_model

print(0)
# Загрузка данных
file_path = 'sent_data_transformed.csv'
dataset, tokenizer = load_bert_data(file_path, max_len=128)
print(1)


# Разделение на обучающую и тестовую выборки
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(2)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
print(3)

# Инициализация модели
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(4)

bert_model = BERTTextErrorClassifier(num_labels=3)
print(5)

# Тренировка модели lr=2e-5
train_bert_model(bert_model.model, train_loader, val_loader, epochs=0, lr=2e-5, device='cpu')
print(6)

# Оценка модели
#evaluate_bert_model(bert_model.model, val_loader, device)
print(7)

save_model(bert_model.model)

# Пример использования модели
new_text = "This is a test sentence for error detection."
predicted_error = bert_model.predict(new_text, device=device)
print(f"Уровень ошибки: {predicted_error}")
