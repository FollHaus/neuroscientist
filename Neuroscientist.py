import os
import requests
import gradio as gr
import tiktoken
import re
from typing import List, Dict, Any

# Альтернативы для LangChain с бесплатными эмбеддингами
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# Конфигурация моделей
models = [
    {
        "doc": "https://docs.google.com/document/d/ВАШ_ID/edit",
        "prompt": '''Ты - виртуальный помощник онлайн-школы, твоя задача быстро, точно и понятно отвечать на вопросы.
                        Говори только по делу, не используй шутки, не проявляй эмоций.
                        Общайся вежливо, но строго. Не выдавай информацию, которой нет в базе.
                        Если вопрос не по теме — направь пользователя в поддержку.
                        ''',
        "name": "Нейро-менеджер онлайн-школы",
        "query": "Как зарегистрироваться?"
    },
]


class LocalGPT:
    """Нейро-сотрудник с использованием локальных моделей"""

    def __init__(self, model="llama2:7b-chat", ollama_url="http://127.0.0.1:11434/"):
        self.log = ''
        self.model = model
        self.ollama_url = ollama_url
        self.search_index = None
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')  # Бесплатная модель эмбеддингов
        self.chroma_client = chromadb.Client()
        self.collection = None

    def load_search_indexes(self, url: str):
        """Загрузка документа из Google Docs"""
        try:
            # Извлечение ID документа
            match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
            if match_ is None:
                raise ValueError('Неверный Google Docs URL')

            doc_id = match_.group(1)

            # Скачивание документа
            response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
            response.raise_for_status()

            text = response.text
            self.log += f'Документ успешно загружен, размер: {len(text)} символов\n'

            return self.create_embedding(text)

        except Exception as e:
            self.log += f'Ошибка загрузки документа: {str(e)}\n'
            return None

    def create_embedding(self, text: str):
        """Создание векторной базы знаний с эмбеддингами"""
        try:
            # Разделение текста на чанки
            chunks = self.split_text(text, chunk_size=1000, overlap=100)

            # Создание коллекции в ChromaDB
            collection_name = f"documents_{len(text)}"
            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass

            self.collection = self.chroma_client.create_collection(collection_name)

            # Создание эмбеддингов для каждого чанка
            embeddings = []
            documents = []
            ids = []

            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Проверяем, что чанк не пустой
                    embedding = self.embeddings_model.encode(chunk)
                    embeddings.append(embedding.tolist())
                    documents.append(chunk)
                    ids.append(f"doc_{i}")

            # Добавление в векторную базу
            if documents:
                self.collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    ids=ids
                )

                self.log += f'Создано {len(documents)} фрагментов документа\n'
                self.log += f'Векторная база данных создана успешно\n'
                return True
            else:
                self.log += 'Не удалось создать фрагменты документа\n'
                return False

        except Exception as e:
            self.log += f'Ошибка создания эмбеддингов: {str(e)}\n'
            return False

    def split_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Разделение текста на чанки"""
        chunks = []
        lines = text.split('\n')
        current_chunk = ""

        for line in lines:
            if len(current_chunk) + len(line) < chunk_size:
                current_chunk += line + '\n'
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = line + '\n'

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def search_similar_docs(self, query: str, k: int = 3) -> List[str]:
        """Поиск похожих документов"""
        if not self.collection:
            return []

        try:
            # Создание эмбеддинга для запроса
            query_embedding = self.embeddings_model.encode(query)

            # Поиск в векторной базе
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k
            )

            return results['documents'][0] if results['documents'] else []

        except Exception as e:
            self.log += f'Ошибка поиска: {str(e)}\n'
            return []

    def query_ollama(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Запрос к локальной модели Ollama"""
        try:
            # Формирование промпта для модели
            prompt = ""
            for message in messages:
                role = message["role"]
                content = message["content"]

                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\nAssistant: "

            # Запрос к Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Нет ответа от модели")
            else:
                self.log += f'Ошибка запроса к Ollama: {response.status_code}\n'
                return "Ошибка подключения к модели"

        except requests.exceptions.ConnectionError:
            return "Ошибка: Ollama не запущен. Запустите 'ollama serve' в терминале"
        except Exception as e:
            self.log += f'Ошибка запроса к модели: {str(e)}\n'
            return f"Ошибка: {str(e)}"

    def answer_index(self, system_prompt: str, query: str, temperature: float = 0.7) -> str:
        """Ответ на основе базы"""
        if not self.collection:
            self.log += 'Модель необходимо обучить!\n'
            return 'Модель не обучена. Загрузите документ для обучения.'

        # Поиск релевантных документов
        docs = self.search_similar_docs(query, k=3)

        if not docs:
            self.log += 'Не найдены релевантные документы\n'
            return 'Извините, не могу найти информацию по вашему запросу в загруженных документах.'

        # Формирование контекста
        context = "\n\n".join([f"Фрагмент {i + 1}:\n{doc}" for i, doc in enumerate(docs)])
        self.log += f'Найдено {len(docs)} релевантных фрагментов\n'

        # Формирование сообщений для модели
        messages = [
            {
                "role": "system",
                "content": system_prompt + f"\n\nКонтекст из документов:\n{context}"
            },
            {
                "role": "user",
                "content": query
            }
        ]

        # Запрос к модели
        response = self.query_ollama(messages, temperature)

        return response


# Функции для Gradio интерфейса
def create_neural_employee(model_config):
    """Создание экземпляра нейро-сотрудника"""
    gpt = LocalGPT()

    # Загрузка документа
    success = gpt.load_search_indexes(model_config["doc"])

    if success:
        # Тестовый запрос
        response = gpt.answer_index(
            model_config["prompt"],
            model_config["query"]
        )

        return f"""
## {model_config["name"]}

**Статус обучения:** ✅ Успешно обучен

**Тестовый вопрос:** {model_config["query"]}

**Ответ:**
{response}
"""
    else:
        return f"""
## {model_config["name"]}

**Статус обучения:** ❌ Ошибка обучения

**Логи ошибок:**
{gpt.log}"""


def chat_with_employee(message, model_config):
    """Чат с нейро-сотрудником"""
    gpt = LocalGPT()
    gpt.load_search_indexes(model_config["doc"])

    response = gpt.answer_index(model_config["prompt"], message)

    return response


# Запуск приложения
if __name__ == "__main__":
    demo = gr.Blocks(title="Нейро-сотрудник онлайн-школы")

    with demo:
        gr.Markdown("""
            # 🤖 Нейро-сотрудник онлайн-школы
            Добро пожаловать! Здесь вы можете обучить модель и задать ей вопросы.
            """)

        with gr.Tab("📘 Обучение модели"):
            gr.Markdown("### Инструкция по обучению модели")
            gr.Markdown("Нажмите кнопку ниже, чтобы загрузить данные и обучить нейро-сотрудника.")
            train_btn = gr.Button("🎓 Обучить модель", variant="primary")
            training_output = gr.Markdown("")

        with gr.Tab("💬 Чат с ботом"):
            gr.Markdown("### Задайте вопрос")
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Например: Не удается войти в аккаунт?",
                    label="Ваш вопрос",
                    lines=2
                )
                chat_btn = gr.Button("Отправить", variant="primary")

            chat_output = gr.Textbox(
                label="Ответ онлайн-ассистента",
                lines=8,
                max_lines=15,
                interactive=False
            )

            gr.Examples(
                examples=[
                    ["Когда начинаются курсы?"],
                    ["Как оплатить обучение?"]
                ],
                inputs=chat_input
            )

        train_btn.click(
            fn=lambda: create_neural_employee(models[0]),
            outputs=training_output
        )

        chat_btn.click(
            fn=lambda msg: chat_with_employee(msg, models[0]),
            inputs=chat_input,
            outputs=chat_output
        )

    demo.launch(
        server_name="localhost",
        server_port=7878,
        share=False
    )
