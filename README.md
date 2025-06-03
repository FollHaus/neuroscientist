# 🧠 Нейро-сотрудник онлайн-школы

Локальный ассистент, который отвечает на вопросы пользователей на основе простой базы знаний из Google Docs.  
Работает на локальной модели (например, LLaMA 2 или Mistral) через [Ollama](https://ollama.com), с векторным поиском по документу.

---

## 🚀 Возможности

- Чтение документа из Google Docs
- Создание векторной базы с бесплатными эмбеддингами (SentenceTransformers)
- Вопросы и ответы с использованием локальной модели через Ollama
- Простой интерфейс через Gradio

---

## 📦 Установка

### 1. Установите зависимости Python:

```bash
pip install -r requirements.txt
```

### 2. Установите и запустите Ollama:

Скачайте и установите Ollama с [https://ollama.com/download](https://ollama.com/download)

Запустите сервер в фоновом режиме:

```bash
ollama serve
```

Загрузите нужную модель (например, LLaMA 2):

```bash
ollama pull llama2:7b-chat
```

---

## 🧑‍🏫 Запуск приложения

```bash
python Neuroscientist.py
```

Откроется веб-интерфейс по адресу [http://localhost:7878](http://localhost:7878)
![image](https://github.com/user-attachments/assets/d45eeaba-0511-4534-b23a-5fef3a9744a7)
![image](https://github.com/user-attachments/assets/bf32d31f-5020-499e-8d92-961343d378ef)
![image](https://github.com/user-attachments/assets/237828f5-e8e4-4700-8864-93f3bc4698b3)

---

## 🛠 Конфигурация

Модель и ссылка на документ указываются в словаре `models`:

```python
models = [
    {
        "doc": "https://docs.google.com/document/d/ВАШ_ID/edit",
        "prompt": "Ваш системный промпт для модели",
        "name": "Название модели",
        "query": "Тестовый вопрос"
    }
]
```
---

## 🧩 Заметки

* Эмбеддинги создаются с помощью `all-MiniLM-L6-v2`, они бесплатные и быстрые.
* Для работы необходим открытый доступ к Google Docs (в режиме чтения).
* Ollama работает локально, интернет не требуется после загрузки модели.
