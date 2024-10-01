import tkinter as tk
from tkinter import messagebox
import torch
from transformers import BertTokenizer
from bert_model import BERTTextErrorClassifier
from load_model import load_model

# Настройка окна
root = tk.Tk()
root.title("Проверка текста на ошибки")
root.geometry("400x300")

# Загрузка модели и токенизатора
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model('bert_model.pth', num_labels=3)
print('works')
model = model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Функция для обработки текста
def check_text():
    text = text_input.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Ошибка", "Пожалуйста, введите текст для проверки!")
        return

    # Токенизация текста
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Предсказание
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, prediction = torch.max(logits, dim=1)

    # Результаты предсказания
    if prediction.item() == 0:
        result_text = "Текст без ошибок."
    elif prediction.item() == 1:
        result_text = "Найдены незначительные ошибки."
    else:
        result_text = "Текст содержит серьезные ошибки."

    messagebox.showinfo("Результат", result_text)

# Ввод текста
text_label = tk.Label(root, text="Введите текст для проверки:", font=("Arial", 12))
text_label.pack(pady=10)

text_input = tk.Text(root, height=10, width=40, font=("Arial", 12))
text_input.pack(pady=10)

# Кнопка для проверки текста
check_button = tk.Button(root, text="Проверить", command=check_text, font=("Arial", 12))
check_button.pack(pady=10)

# Запуск приложения
root.mainloop()
