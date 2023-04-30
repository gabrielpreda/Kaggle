#-------------------------------------
#
# Audio plot utils:
#   - text formater (auxiliar/utils)
#   - sound player
#   - display sound wave
#   - display spectrogram
#   - display mel spectrogram
#
#-------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa.display import waveshow
import IPython.display as ipd
from IPython.display import Audio 
from IPython.display import display, HTML


def cstr(str_text, color='black'):
    """
    Html styling for widgets
    Args
        str_text: text to disply
        color: color to display the text
    Returns
        Formated text/label
    """
    return "<text style=color:{}><strong>{}<strong></text>".format(color, str_text)

def play_sound(sound_path="",
               text="Test", 
               color="green"):
    """
    Display a sound play widget
    Args
        sound_path: path to the sound file
        text: text to display
        color: color for text to display
    Returns
        None
    """
    display(HTML(cstr(text, color)))
    display(ipd.Audio(sound_path))
    
def display_sound_wave(sound_path=None,
               text="Test", 
               color="green"):
    """
    Display a sound wave
    Args
        sound_path: path to the sound file
        text: text to display
        color: color for text to display
    Returns
        None
    """
    if not sound_path:
        return    
    y_sound, sr_sound = librosa.load(sound_path)
    audio_sound, _ = librosa.effects.trim(y_sound)
    fig, ax = plt.subplots(1, figsize = (16, 3))
    fig.suptitle(f'Sound Wave: {text}', fontsize=12)
    librosa.display.waveshow(y = audio_sound, sr = sr_sound, color = color)
    
    
def display_spectrogram(sound_path=None,
               text="Test"):
    """
    Display a spectrogram
    Args
        sound_path: path to the sound file
        text: text to display (title)
    Returns
        None    
    """
    if not sound_path:
        return
    n_fft=2048
    hop_length=512
    y_sound, sr_sound = librosa.load(sound_path)
    audio_sound, _ = librosa.effects.trim(y_sound)
    
    # Short-time Fourier transform (STFT)
    D_sound = np.abs(librosa.stft(audio_sound, n_fft = n_fft, hop_length = hop_length))
    # Convert an amplitude spectrogram to Decibels-scaled spectrogram.
    DB_sound = librosa.amplitude_to_db(D_sound, ref = np.max)
    # Prepare the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    img=librosa.display.specshow(DB_sound, sr = sr_sound, hop_length = hop_length, x_axis = 'time', 
                             y_axis = 'log', cmap = 'cool', ax=ax)
    ax.set_title(f'Log Frequency Spectrogram: {text}', fontsize=10) 
    plt.colorbar(img,ax=ax)
    
def display_mel_spectrogram(sound_path=None,
               text="Test"):
    """
    Display a mel spectrogram
    Args
        sound_path: path to the sound file
        text: text to display (title)
    Returns
        None    
    """    
    if not sound_path:
        return
    hop_length=512
    y_sound, sr_sound = librosa.load(sound_path)
    audio_sound, _ = librosa.effects.trim(y_sound)
    # Create the Mel Spectrograms
    S_sound = librosa.feature.melspectrogram(audio_sound, sr=sr_sound)
    S_DB_sound = librosa.amplitude_to_db(S_sound, ref=np.max)
    # Prepare the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    img=librosa.display.specshow(S_DB_sound, sr = sr_sound, hop_length = hop_length, x_axis = 'time', 
                             y_axis = 'log', cmap = 'cool', ax=ax)

    ax.set_title(f'Mel Spectrogram: {text}', fontsize=10) 
    plt.colorbar(img,ax=ax)
