import os
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import which


class AudioProcessor:
    """
    Audio processing class for converting audio files to text
    """
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Check if ffmpeg is available
        if not which("ffmpeg"):
            print("Warning: ffmpeg not found. Audio conversion may not work properly.")
    
    def convert_audio_to_wav(self, audio_file_path):
        """
        Convert audio file to WAV format for speech recognition
        """
        try:
            # Load audio file
            audio = AudioSegment.from_file(audio_file_path)
            
            # Convert to mono and set sample rate
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                audio.export(temp_wav.name, format="wav")
                return temp_wav.name
                
        except Exception as e:
            raise Exception(f"Error converting audio: {str(e)}")
    
    def transcribe_audio(self, audio_file_path):
        """
        Convert audio file to text using speech recognition
        """
        wav_path = None
        try:
            # Convert to WAV if needed
            if not audio_file_path.lower().endswith('.wav'):
                wav_path = self.convert_audio_to_wav(audio_file_path)
            else:
                wav_path = audio_file_path
            
            # Transcribe audio
            with sr.AudioFile(wav_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                # Record the audio
                audio_data = self.recognizer.record(source)
                
                # Convert to text using Google Speech Recognition
                try:
                    text = self.recognizer.recognize_google(audio_data)
                    return text
                except sr.UnknownValueError:
                    raise Exception("Could not understand the audio. Please try with clearer audio.")
                except sr.RequestError as e:
                    raise Exception(f"Speech recognition service error: {str(e)}")
                    
        except Exception as e:
            raise Exception(f"Error transcribing audio: {str(e)}")
            
        finally:
            # Clean up temporary file
            if wav_path and wav_path != audio_file_path and os.path.exists(wav_path):
                os.unlink(wav_path)
    
    def process_audio_file(self, file_stream, filename):
        """
        Process uploaded audio file and return transcribed text
        """
        temp_path = None
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as temp_file:
                file_stream.save(temp_file.name)
                temp_path = temp_file.name
            
            # Transcribe the audio
            transcribed_text = self.transcribe_audio(temp_path)
            return transcribed_text
            
        except Exception as e:
            raise Exception(f"Error processing audio file: {str(e)}")
            
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
