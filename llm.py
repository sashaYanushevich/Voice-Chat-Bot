import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables (e.g., OPENROUTER_API_KEY)
load_dotenv()

class OpenRouterClient:
    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct"):
        """
        Инициализирует клиент OpenRouter с моделью по умолчанию.
        """
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.model = model
        self.conversation_history = []

    async def chat_completion(self, user_message: str, system_message: str = None) -> str:
        """
        Отправляет запрос к OpenRouter API и возвращает ответ.
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Добавляем историю разговора
        messages.extend(self.conversation_history)
        
        # Добавляем новое сообщение пользователя
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            assistant_message = response.choices[0].message.content
            
            # Сохраняем в историю разговора
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Ограничиваем историю последними 10 сообщениями (5 пар)
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return assistant_message
            
        except Exception as e:
            return f"Ошибка при обращении к API: {str(e)}"

    async def stream_completion(self, user_message: str, system_message: str = None):
        """
        Потоковая генерация ответа.
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )
            
            assistant_message = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    assistant_message += content
                    yield content
            
            # Сохраняем полный ответ в историю
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
                
        except Exception as e:
            yield f"Ошибка при обращении к API: {str(e)}"

    def clear_history(self):
        """Очищает историю разговора."""
        self.conversation_history = []

async def run_batch_example():
    """
    Пример использования в batch режиме.
    """
    client = OpenRouterClient()
    user_input = input("Введите ваш вопрос: ").strip()
    
    system_prompt = "Ты полезный AI-ассистент, который отвечает лаконично и по делу."
    
    response = await client.chat_completion(user_input, system_prompt)
    print(f"\nResponse: {response}")



