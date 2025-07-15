import asyncio
import io
import os
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
    PrerecordedOptions,
)

# Load environment variables (e.g., Deepgram API key from .env file)
load_dotenv()

class TranscriptManager:
    """
    Collects and manages transcript fragments during streaming.
    """
    def __init__(self):
        self.fragments = []

    def add_fragment(self, text: str):
        self.fragments.append(text)

    def get_combined_transcript(self) -> str:
        return ' '.join(self.fragments).strip()

    def reset(self):
        self.fragments.clear()


class DeepgramSTT:
    """
    Deepgram Speech-to-Text client для API интеграции.
    """
    
    def __init__(self):
        """Инициализация Deepgram STT клиента."""
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not found in environment variables")
        
        client_config = DeepgramClientOptions(options={"keepalive": "true"})
        self.client = DeepgramClient(self.api_key, client_config)
        
        # Улучшенные настройки для предзаписанного аудио
        self.prerecorded_options = PrerecordedOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            smart_format=True,
            # Улучшения для коротких фраз и предотвращения обрезания первого слова
            diarize=False,  # Отключаем диаризацию для лучшей производительности
            utterances=True,  # Включаем разделение на высказывания
            paragraphs=False,  # Отключаем для коротких фраз
            detect_language=False,  # Отключаем автоопределение языка
            # Настройки для улучшения качества и предотвращения обрезания
            profanity_filter=False,
            redact=False,
            search=None,
            replace=None,
            keywords=None,
            version="latest",
            # Дополнительные настройки для лучшего захвата начала речи
            multichannel=False,
            alternatives=1,
            numerals=True
        )
        
        # Альтернативные настройки для очень коротких фраз (используем nova-2 с другими параметрами)
        self.short_phrase_options = PrerecordedOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            smart_format=True,
            utterances=True,
            detect_language=False,
            # Более мягкие настройки для коротких фраз чтобы не обрезать первое слово
            filler_words=False,  # Убираем слова-паразиты
            profanity_filter=False,
            redact=False,
            # Дополнительные настройки для лучшего захвата коротких фраз
            multichannel=False,
            alternatives=1,
            numerals=True,
            diarize=False
        )
    
    async def transcribe_audio_bytes(self, audio_bytes: bytes) -> str:
        """
        Улучшенная транскрипция аудио из байтов с поддержкой коротких фраз.
        
        Args:
            audio_bytes: Аудио данные в байтах
            
        Returns:
            str: Распознанный текст
        """
        try:
            # Определяем длительность аудио для выбора оптимальной стратегии
            audio_duration = self._estimate_audio_duration(audio_bytes)
            print(f"🎧 Estimated audio duration: {audio_duration:.2f}s")
            
            # Создаем источник аудио из байтов
            audio_source = {"buffer": audio_bytes}
            
            # Пробуем различные конфигурации в порядке приоритета
            configurations = [
                ("short_phrase", self.short_phrase_options),
                ("standard", self.prerecorded_options),
                ("minimal", self._get_minimal_options())
            ]
            
            transcript = ""
            for config_name, options in configurations:
                print(f"🔄 Trying {config_name} configuration...")
                transcript = await self._try_transcription(audio_source, options)
                
                if transcript:
                    print(f"✅ Success with {config_name} configuration")
                    break
                else:
                    print(f"⚠️ {config_name} configuration failed, trying next...")
            
            # Постобработка результата
            if transcript:
                transcript = self._post_process_transcript(transcript)
                print(f"✅ Final STT result: '{transcript}'")
            else:
                print("⚠️ All transcription attempts failed")
            
            return transcript
            
        except Exception as e:
            print(f"❌ STT transcription error: {e}")
            return ""
    
    def _get_minimal_options(self) -> PrerecordedOptions:
        """
        Минимальные настройки для максимальной совместимости.
        """
        return PrerecordedOptions(
            model="nova-2",
            language="en-US",
            punctuate=True
        )
    
    def _estimate_audio_duration(self, audio_bytes: bytes) -> float:
        """
        Приблизительная оценка длительности аудио.
        Простая эвристика на основе размера файла.
        """
        # Примерная оценка: WebM Opus ~16kbps для речи
        estimated_duration = len(audio_bytes) / (16000 / 8)  # байт/сек
        return max(0.1, min(estimated_duration, 30.0))  # Ограничиваем от 0.1 до 30 сек
    
    
    async def _try_transcription(self, audio_source: dict, options: PrerecordedOptions) -> str:
        """
        Выполняет транскрипцию с заданными настройками.
        """
        try:
            response = await self.client.listen.asyncprerecorded.v("1").transcribe_file(
                audio_source,
                options
            )
            
            # Извлекаем текст из ответа
            if (response.results and
                response.results.channels and
                len(response.results.channels) > 0 and
                response.results.channels[0].alternatives and
                len(response.results.channels[0].alternatives) > 0):
                
                transcript = response.results.channels[0].alternatives[0].transcript
                return transcript.strip()
            
            return ""
            
        except Exception as e:
            print(f"⚠️ Transcription attempt failed: {e}")
            return ""
    
    def _post_process_transcript(self, transcript: str) -> str:
        """
        Постобработка транскрипта для улучшения качества.
        """
        if not transcript:
            return ""
        
        # Убираем лишние пробелы
        transcript = ' '.join(transcript.split())
        
        # Проверяем и исправляем обрезанные первые слова
        transcript = self._fix_truncated_first_word(transcript)
        
        # Исправляем распространенные ошибки для коротких фраз
        corrections = {
            # Распространенные ошибки в коротких ответах
            'yeah': 'yes',
            'yep': 'yes',
            'nope': 'no',
            'uh huh': 'yes',
            'mm hmm': 'yes',
            'uh uh': 'no',
            # Исправления для технических терминов
            'react': 'React',
            'javascript': 'JavaScript',
            'typescript': 'TypeScript',
            'node': 'Node',
            'angular': 'Angular',
            'vue': 'Vue'
        }
        
        # Применяем исправления только для коротких фраз (до 5 слов)
        words = transcript.split()
        if len(words) <= 5:
            transcript_lower = transcript.lower()
            for wrong, correct in corrections.items():
                if transcript_lower == wrong or transcript_lower.startswith(wrong + ' ') or transcript_lower.endswith(' ' + wrong):
                    transcript = transcript_lower.replace(wrong, correct)
                    break
        
        return transcript
    
    def _fix_truncated_first_word(self, transcript: str) -> str:
        """
        Исправляет обрезанные первые слова в транскрипте.
        """
        if not transcript:
            return transcript
        
        # Словарь для исправления обрезанных слов
        truncation_fixes = {
            # Обрезанные приветствия
        
            'eact': 'React',
            'avaScript': 'JavaScript',
            'ypeScript': 'TypeScript',
            'ode': 'Node',
            'ngular': 'Angular',
            'ue': 'Vue'
        }
        
        words = transcript.split()
        if words:
            first_word = words[0].lower()
            
            # Проверяем точные совпадения
            if first_word in truncation_fixes:
                words[0] = truncation_fixes[first_word]
                return ' '.join(words)
            
            # Проверяем частичные совпадения для коротких слов (до 4 символов)
            if len(first_word) <= 4:
                for truncated, full in truncation_fixes.items():
                    if first_word == truncated.lower():
                        words[0] = full
                        return ' '.join(words)
        
        return transcript
    
    async def transcribe_audio_file(self, file_path: str) -> str:
        """
        Транскрибирует аудио файл.
        
        Args:
            file_path: Путь к аудио файлу
            
        Returns:
            str: Распознанный текст
        """
        try:
            with open(file_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                return await self.transcribe_audio_bytes(audio_bytes)
                
        except Exception as e:
            print(f"STT file transcription error: {e}")
            return ""

async def transcribe_from_microphone():
    """
    Streams audio from the microphone to Deepgram and prints transcripts live.
    """
    try:
        client_config = DeepgramClientOptions(options={"keepalive": "true"})
        dg_client = DeepgramClient("", client_config)
        dg_stream = dg_client.listen.asynclive.v("1")

        transcript_manager = TranscriptManager()
        terminate_event = asyncio.Event()

        async def handle_transcript(_, result, **kwargs):
            if not result.channel.alternatives:
                return
            text = result.channel.alternatives[0].transcript.strip()

            if not text:
                return  # Skip empty transcripts

            if not result.speech_final:
                transcript_manager.add_fragment(text)
            else:
                transcript_manager.add_fragment(text)
                full_text = transcript_manager.get_combined_transcript()
                if full_text:  # Only print if there's actual speech
                    print(f"Speaker: {full_text}")

                    if "goodbye" in full_text.lower():
                        terminate_event.set()
                transcript_manager.reset()

        async def handle_error(_, error, **kwargs):
            print(f"[Error] {error}")

        dg_stream.on(LiveTranscriptionEvents.Transcript, handle_transcript)
        dg_stream.on(LiveTranscriptionEvents.Error, handle_error)

        # Улучшенные настройки для живой транскрипции коротких фраз
        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            # Настройки для лучшего распознавания коротких фраз
            endpointing=500,  # Увеличиваем время ожидания окончания речи (мс) - предотвращает обрезание первого слова
            vad_events=True,  # Включаем события детекции голоса
            interim_results=True,  # Включаем промежуточные результаты
            utterance_end_ms=1500,  # Увеличиваем время тишины для завершения высказывания
            vad_turnoff=250,  # Задержка перед отключением VAD - помогает захватить начало речи
            smart_format=True,  # Умное форматирование
            profanity_filter=False,
            redact=False,
            diarize=False,  # Отключаем для лучшей производительности
            multichannel=False,
            alternatives=1,  # Получаем только лучший результат
            numerals=True,  # Преобразуем числа в цифры
            search=None,
            replace=None,
            keywords=None
        )

        await dg_stream.start(options)

        mic = Microphone(dg_stream.send)
        mic.start()
        
        print()
        print("Listening.......")
        print("Say 'goodbye' to end the session.")
        print()

        while mic.is_active() and not terminate_event.is_set():
            await asyncio.sleep(0.5)

        mic.finish()
        await dg_stream.finish()
        print("Session ended.")

    except Exception as ex:
        print(f"An error occurred: {ex}")

if __name__ == "__main__":
    asyncio.run(transcribe_from_microphone())
