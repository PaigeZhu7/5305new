"""Main module that classifies the emotion in a live-recorded voice."""
from voice_recorder import record_voice
from predictions import make_predictions
import os
from config import features_PATH  
from config import images_PATH  
from config import models_PATH  
from config import recordings_PATH  

def classify_myvoice():
    record_voice()   
    myvoice_path = os.path.join(recordings_PATH, 'myvoice.wav')
    make_predictions(myvoice_path)



if __name__ == "__main__":
    classify_myvoice()
