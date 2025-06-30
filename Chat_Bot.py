import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

class LanguageModelProcessor:
    def __init__(self):
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY не найден в переменных окружения.")

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        
        self.model = "meta-llama/llama-3.1-70b-instruct"
        self.conversation_history = []

        # Загружаем системный промпт
        try:
            with open('Bot_prompt.txt', 'r', encoding='utf-8') as file:
                self.system_prompt = file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError("Файл 'Bot_prompt.txt' не найден.")

    async def process(self, text: str) -> str:
        """Обрабатывает входящий текст и возвращает ответ от LLM."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Добавляем историю разговора
        messages.extend(self.conversation_history)
        
        # Добавляем новое сообщение пользователя
        messages.append({"role": "user", "content": text})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            assistant_message = response.choices[0].message.content
            
            # Сохраняем в историю разговора
            self.conversation_history.append({"role": "user", "content": text})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Ограничиваем историю последними 20 сообщениями (10 пар)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return assistant_message
            
        except Exception as e:
            return f"Ошибка при обращении к API: {str(e)}"

class ConversationManager:
    def __init__(self):
        self.llm_processor = LanguageModelProcessor()
        print()
        print("🤖 Chat bot initialized. Type 'goodbye' to exit.")
        print()

    async def main(self):
        """Основной цикл разговора."""
        while True:
            user_input = input("Вы: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "goodbye"]:
                print("🤖: Goodbye!")
                break

            try:
                llm_response = await self.llm_processor.process(user_input)
                print(f"🤖: {llm_response}")
                print()
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())
