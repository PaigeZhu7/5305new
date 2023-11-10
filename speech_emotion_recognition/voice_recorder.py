import soundfile as sf
import sounddevice as sd
from scipy.io.wavfile import write
import os
from config import features_PATH  
from config import images_PATH  
from config import models_PATH  
from config import recordings_PATH  

def record_voice():
    """This function records your voice and saves the output as .wav file."""
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording
    # sd.default.device = "Built-in Audio"  # Speakers full name here

    print("Say something:")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    myvoice_path = os.path.join(recordings_PATH, 'myvoice.wav')
    write(myvoice_path, fs, myrecording)
    print("Voice recording saved.")
