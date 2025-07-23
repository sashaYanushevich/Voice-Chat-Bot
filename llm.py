import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables (e.g., OPENROUTER_API_KEY)
load_dotenv()

class OpenRouterClient:
    def __init__(self, model: str = "openai/gpt-4.1"):
        """
        Инициализирует клиент OpenRouter с моделью по умолчанию.
        """
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.model = model
        self.conversation_history = []
        self.interview_completed = False  # Флаг завершения интервью

    async def chat_completion(self, user_message: str, system_message: str = None) -> tuple[str, bool]:
        """
        Отправляет запрос к OpenRouter API и возвращает ответ с флагом завершения.
        Возвращает: (response_text, is_interview_ended)
        """
        # Проверяем, завершено ли интервью
        if self.interview_completed:
            return "Thank you for completing the screening interview. Our recruitment team will be in touch soon to discuss the next steps. Enjoy the rest of your day!", True
        
        messages = []
        
        if system_message:
            # Добавляем инструкции для метки завершения в системный промпт
            enhanced_system_message = f"""{system_message}

INTERVIEW COMPLETION INSTRUCTIONS:
When you have finished asking all your interview questions and are ready to end the interview, you MUST include the special marker [INTERVIEW_END] at the very beginning of your final response.

Example: "[INTERVIEW_END] Thank you for completing the screening interview. Our recruitment team will be in touch soon to discuss the next steps. Enjoy the rest of your day!"

This marker will signal the system to automatically end the interview session."""
            messages.append({"role": "system", "content": enhanced_system_message})
        
        # Добавляем историю разговора
        messages.extend(self.conversation_history)
        
        # Добавляем новое сообщение пользователя
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            assistant_message = response.choices[0].message.content
            
            # Проверяем наличие метки завершения интервью
            interview_ended = False
            if assistant_message.startswith("[INTERVIEW_END]"):
                interview_ended = True
                self.interview_completed = True
                # Убираем метку из сообщения для пользователя
                assistant_message = assistant_message.replace("[INTERVIEW_END]", "").strip()
                print("🎯 Interview completion marker detected!")
            
            # Сохраняем в историю разговора
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Ограничиваем историю последними 10 сообщениями (5 пар)
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return assistant_message, interview_ended
            
        except Exception as e:
            return f"Ошибка при обращении к API: {str(e)}", False

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
                max_tokens=500,
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
        self.interview_completed = False
    
    def is_interview_completed(self) -> bool:
        """Проверяет, завершено ли интервью."""
        return self.interview_completed
    
    def get_interview_status(self) -> dict:
        """Возвращает статус интервью."""
        return {
            'completed': self.interview_completed,
            'conversation_length': len(self.conversation_history)
        }

async def run_batch_example():
    """
    Пример использования в batch режиме.
    """
    client = OpenRouterClient()
    user_input = input("Введите ваш вопрос: ").strip()
    
    system_prompt = "Ты полезный AI-ассистент, который отвечает лаконично и по делу."
    
    response = await client.chat_completion(user_input, system_prompt)
    print(f"\nResponse: {response}")



