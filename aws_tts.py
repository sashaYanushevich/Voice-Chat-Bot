import asyncio
import re
import subprocess
import time
from typing import List, Optional
import io
import os

import boto3
from botocore.exceptions import ClientError, NoCredentialsError


class AWSPollyTTS:
    """AWS Polly TTS с chunking и асинхронной обработкой."""
    
    def __init__(self, 
                 voice_id: str = "Joanna",
                 region_name: str = "us-east-1",
                 chunk_size: int = 300):
        """
        Args:
            voice_id: Голос AWS Polly (Joanna, Matthew, Salli и т.д.)
            region_name: AWS регион
            chunk_size: Максимальный размер чанка в символах
        """
        self.voice_id = voice_id
        self.chunk_size = chunk_size
        
        try:
            self.polly_client = boto3.client('polly', region_name=region_name)
            # Тестируем подключение
            self.polly_client.describe_voices()
        except (ClientError, NoCredentialsError) as e:
            print(f"❌ AWS Polly error: {e}")
            print("💡 Make sure AWS credentials are configured:")
            print("   aws configure")
            print("   or set environment variables:")
            print("   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            raise
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Разбивает текст на чанки по предложениям.
        """
        # Разбиваем по предложениям (учитываем русский и английский)
        sentences = re.split(r'[.!?]+\s*', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Если предложение само по себе слишком длинное
            if len(sentence) > self.chunk_size:
                # Добавляем текущий чанк если он есть
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Разбиваем длинное предложение по запятым
                parts = sentence.split(', ')
                for part in parts:
                    if len(current_chunk + part) > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part + ", "
                    else:
                        current_chunk += part + ", "
            else:
                # Проверяем поместится ли предложение в текущий чанк
                if len(current_chunk + sentence) > self.chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
                else:
                    current_chunk += sentence + ". "
        
        # Добавляем последний чанк
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def synthesize_chunk(self, text: str) -> Optional[bytes]:
        """
        Синтезирует один чанк текста в аудио.
        """
        try:
            # Выполняем синтез в отдельном потоке чтобы не блокировать event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.polly_client.synthesize_speech(
                    Text=text,
                    OutputFormat='mp3',
                    VoiceId=self.voice_id,
                    Engine='neural'  # Используем neural engine для лучшего качества
                )
            )
            
            # Читаем аудио данные
            audio_data = response['AudioStream'].read()
            return audio_data
            
        except Exception as e:
            print(f"❌ Chunk synthesis error: {e}")
            return None
    
    def play_audio_chunk(self, audio_data: bytes):
        """
        Воспроизводит аудио чанк через ffplay.
        """
        try:
            player_command = ["ffplay", "-autoexit", "-", "-nodisp", "-loglevel", "quiet"]
            player_process = subprocess.Popen(
                player_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            
            player_process.stdin.write(audio_data)
            player_process.stdin.close()
            player_process.wait()
            
        except Exception as e:
            print(f"❌ Playback error: {e}")
    
    async def speak(self, text: str):
        """
        Главный метод: говорит текст с chunking и параллельной обработкой.
        """
        if not text.strip():
            return
        
        start_time = time.time()
        
        # Разбиваем текст на чанки
        chunks = self.split_text_into_chunks(text)
        print(f"🎤 Split into {len(chunks)} chunks")
        
        # Создаём очередь для аудио чанков
        audio_queue = asyncio.Queue(maxsize=3)  # Буферизуем до 3 чанков
        
        # Флаг завершения синтеза
        synthesis_done = asyncio.Event()
        
        async def synthesizer():
            """Корутин для синтеза чанков."""
            try:
                for i, chunk in enumerate(chunks):
                    print(f"🔄 Synthesizing chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
                    audio_data = await self.synthesize_chunk(chunk)
                    if audio_data:
                        await audio_queue.put(audio_data)
                    else:
                        print(f"⚠️  Skipped chunk {i+1}")
            except Exception as e:
                print(f"❌ Synthesizer error: {e}")
            finally:
                synthesis_done.set()
        
        async def player():
            """Корутин для воспроизведения чанков."""
            try:
                chunk_count = 0
                while True:
                    try:
                        # Ждём следующий чанк с таймаутом
                        audio_data = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                        chunk_count += 1
                        
                        print(f"🔊 Playing chunk {chunk_count}")
                        
                        # Воспроизводим в отдельном потоке
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, self.play_audio_chunk, audio_data)
                        
                        audio_queue.task_done()
                        
                    except asyncio.TimeoutError:
                        # Если очередь пуста и синтез завершён - выходим
                        if synthesis_done.is_set() and audio_queue.empty():
                            break
                        continue
                        
            except Exception as e:
                print(f"❌ Player error: {e}")
        
        # Запускаем синтез и воспроизведение параллельно
        synthesis_task = asyncio.create_task(synthesizer())
        playback_task = asyncio.create_task(player())
        
        # Ждём завершения обеих задач
        await asyncio.gather(synthesis_task, playback_task)
        
        total_time = time.time() - start_time
        print(f"✅ TTS завершён за {total_time:.1f}с ({len(chunks)} чанков)")

# Тестирование (если запускается напрямую)
async def test_polly():
    """Тестирование AWS Polly TTS."""
    try:
        tts = AWSPollyTTS(voice_id="Joanna")
        
        test_text = """
        Привет! Это тестирование AWS Polly с chunking технологией. 
        Каждое предложение обрабатывается отдельно. 
        Пока одно воспроизводится, другие уже синтезируются в фоне.
        Это даёт намного лучшую производительность и меньшую задержку!
        """
        
        print("🎤 Начинаем тест AWS Polly...")
        await tts.speak(test_text)
        print("🎉 Тест завершён!")
        
    except Exception as e:
        print(f"❌ Ошибка теста: {e}")

if __name__ == "__main__":
    asyncio.run(test_polly()) 