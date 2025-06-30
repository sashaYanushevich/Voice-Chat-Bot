import asyncio
import os
import shutil
import subprocess
import sys
import time
from typing import Optional

import requests
from dotenv import load_dotenv
from deepgram import (DeepgramClient, DeepgramClientOptions, LiveOptions,
                     LiveTranscriptionEvents, Microphone)
from openai import AsyncOpenAI

load_dotenv()

class Config:
    """Загружает и валидирует конфигурацию из переменных окружения."""
    def __init__(self):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        
        # Модели
        self.llm_model = "meta-llama/llama-3.1-70b-instruct"
        self.stt_model = "nova-2"
        self.tts_model = "aura-helios-en"
        
        # Пути файлов
        self.bot_prompt_file = "Bot_prompt.txt"
        
        self._validate()

    def _validate(self):
        """Проверяет наличие всех необходимых конфигураций."""
        if not self.openrouter_api_key:
            raise ValueError("ОШИБКА: OPENROUTER_API_KEY не установлен в переменных окружения.")
        if not self.deepgram_api_key:
            raise ValueError("ОШИБКА: DEEPGRAM_API_KEY не установлен в переменных окружения.")
        if not os.path.exists(self.bot_prompt_file):
            raise FileNotFoundError(f"ОШИБКА: Файл промпта не найден: '{self.bot_prompt_file}'")
        if not self._is_installed("ffplay"):
            raise RuntimeError("ОШИБКА: ffplay не установлен. Установите ffmpeg для воспроизведения аудио.")

    @staticmethod
    def _is_installed(lib_name: str) -> bool:
        """Проверяет доступность утилиты командной строки."""
        return shutil.which(lib_name) is not None

# --- Классы сервисов ---

class LiveTranscriber:
    """Обрабатывает транскрипцию речи в реальном времени с помощью Deepgram."""
    def __init__(self, config: Config):
        client_config = DeepgramClientOptions(options={"keepalive": "true"})
        self.client = DeepgramClient(config.deepgram_api_key, client_config)
        self.stt_model = config.stt_model
        self.transcript_future: Optional[asyncio.Future] = None

    async def listen(self) -> str:
        """
        Слушает одно полное предложение с микрофона и возвращает его.
        """
        self.transcript_future = asyncio.Future()
        
        connection = self.client.listen.asynclive.v("1")
        connection.on(LiveTranscriptionEvents.Transcript, self._on_message)
        connection.on(LiveTranscriptionEvents.Error, self._on_error)

        options = LiveOptions(
            model=self.stt_model,
            language="en-US",
            punctuate=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=200,  # Уменьшаем время тишины для лучшей отзывчивости
            smart_format=True,
            interim_results=False,
        )
        await connection.start(options)
        
        microphone = Microphone(connection.send)
        microphone.start()

        try:
            final_transcript = await self.transcript_future
            return final_transcript
        finally:
            microphone.finish()
            await connection.finish()

    async def _on_message(self, _, result, **kwargs):
        """Обратный вызов для обработки сообщений транскрипции от Deepgram."""
        if result.is_final and result.channel.alternatives[0].transcript.strip():
            transcript = result.channel.alternatives[0].transcript
            if self.transcript_future and not self.transcript_future.done():
                self.transcript_future.set_result(transcript)

    async def _on_error(self, _, error, **kwargs):
        """Обратный вызов для обработки ошибок соединения."""
        print(f"\nSTT Error: {error}\n")
        if self.transcript_future and not self.transcript_future.done():
            self.transcript_future.set_exception(Exception(f"Ошибка STT: {error}"))

class LLMProcessor:
    """Управляет взаимодействием с языковой моделью через OpenRouter."""
    def __init__(self, config: Config):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.openrouter_api_key,
        )
        self.model = config.llm_model
        self.conversation_history = []
        
        # Загружаем системный промпт
        with open(config.bot_prompt_file, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read().strip()

    async def generate_response(self, user_text: str) -> str:
        """Генерирует ответ от LLM на основе пользовательского ввода."""
        start_time = time.time()
        
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Добавляем историю разговора
        messages.extend(self.conversation_history)
        
        # Добавляем новое сообщение пользователя
        messages.append({"role": "user", "content": user_text})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500  # Ограничиваем для голосового ответа
            )
            
            ai_response = response.choices[0].message.content
            
            # Сохраняем в историю разговора
            self.conversation_history.append({"role": "user", "content": user_text})
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            # Ограничиваем историю последними 20 сообщениями (10 пар)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            end_time = time.time()
            elapsed_ms = int((end_time - start_time) * 1000)
            
            print(f"LLM ({elapsed_ms}ms): {ai_response}")
            return ai_response
            
        except Exception as e:
            error_msg = f"Ошибка при обращении к LLM: {str(e)}"
            print(error_msg)
            return "Извините, произошла ошибка при обработке вашего запроса."

class SpeechSynthesizer:
    """Обрабатывает преобразование текста в речь с помощью AWS Polly."""
    def __init__(self, config: Config):
        # Импортируем AWS TTS
        from aws_tts import AWSPollyTTS
        
        # Настройки для AWS Polly
        voice_id = getattr(config, 'aws_voice_id', 'Salli')  # или 'Matthew', 'Salli'
        region = getattr(config, 'aws_region', 'us-east-1')
        chunk_size = getattr(config, 'tts_chunk_size', 30)
        
        try:
            self.tts = AWSPollyTTS(
                voice_id=voice_id,
                region_name=region,
                chunk_size=chunk_size
            )
        except Exception as e:
            print(f"❌ AWS Polly initialization error: {e}")
            print("💡 Fallback to Deepgram TTS...")
            # Fallback на старый Deepgram TTS
            self._init_deepgram_fallback(config)

    def _init_deepgram_fallback(self, config):
        """Fallback на Deepgram TTS если AWS Polly недоступен."""
        self.tts = None
        self.api_key = config.deepgram_api_key
        self.model_name = config.tts_model
        self.api_url = f"https://api.deepgram.com/v1/speak?model={self.model_name}&encoding=linear16&sample_rate=24000"

    async def speak(self, text: str):
        """Преобразует текст в речь и воспроизводит."""
        if hasattr(self, 'tts') and self.tts:
            # Используем AWS Polly с chunking
            await self.tts.speak(text)
        else:
            # Fallback на Deepgram
            await self._speak_deepgram(text)

    async def _speak_deepgram(self, text: str):
        """Fallback метод с Deepgram TTS."""
        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"}
        payload = {"text": text}
        
        player_command = ["ffplay", "-autoexit", "-", "-nodisp", "-loglevel", "quiet"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        request_start_time = time.time()
        
        try:
            import requests
            with requests.post(self.api_url, stream=True, headers=headers, json=payload, timeout=20) as response:
                response.raise_for_status()
                first_byte_received = False
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        if not first_byte_received:
                            ttfb = int((time.time() - request_start_time) * 1000)
                            print(f"TTS TTFB: {ttfb}ms")
                            first_byte_received = True
                        player_process.stdin.write(chunk)
                        player_process.stdin.flush()
        except Exception as e:
            print(f"Deepgram TTS error: {e}")
        finally:
            if player_process.stdin:
                player_process.stdin.close()
            player_process.wait()

# --- Главный оркестратор приложения ---

class VoiceAssistant:
    """Главный класс приложения, который управляет потоком разговора."""
    TERMINATION_PHRASES = ["goodbye", "exit", "quit", "stop", "bye"]

    def __init__(self, config: Config):
        self.transcriber = LiveTranscriber(config)
        self.llm_processor = LLMProcessor(config)
        self.synthesizer = SpeechSynthesizer(config)

    async def run(self):
        """Главный цикл голосового ассистента."""
        print("--- 🎤 Voice Assistant Activated ---")
        print(f"Say any of these phrases to exit: {', '.join(self.TERMINATION_PHRASES)}")
        
        while True:
            try:
                print("\n🎧 Listening...")
                user_text = await self.transcriber.listen()
                
                if not user_text:
                    continue
                    
                print(f"👤 Human: {user_text}")

                # Check for termination phrases
                if any(phrase in user_text.lower().strip() for phrase in self.TERMINATION_PHRASES):
                    print("Termination phrase detected. Shutting down.")
                    goodbye_message = "Goodbye! Have a great day!"
                    print(f"🤖 AI: {goodbye_message}")
                    await self.synthesizer.speak(goodbye_message)
                    break

                ai_response = await self.llm_processor.generate_response(user_text)
                await self.synthesizer.speak(ai_response)
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                print("Restarting listening loop...")
                await asyncio.sleep(1)

async def main():
    """Инициализирует и запускает голосового ассистента."""
    try:
        config = Config()
        assistant = VoiceAssistant(config)
        await assistant.run()
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n--- 🛑 Assistant deactivated by user ---")
    except Exception as e:
        print(f"Unexpected critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
