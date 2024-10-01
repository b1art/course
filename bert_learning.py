from transformers import AdamW
from torch.optim import lr_scheduler
import torch
import torch.nn.functional as F

def train_bert_model(model, train_loader, val_loader, epochs, lr, device):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Шаговое снижение скорости обучения
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Извлекаем logits из SequenceClassifierOutput
            logits = outputs.logits

            # Рассчитываем потери с помощью CrossEntropyLoss
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()  # Обновляем learning rate
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')

def evaluate_bert_model(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Рассчитываем потери для валидации
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')
    print(f'Validation Loss: {total_loss / len(val_loader)}')
