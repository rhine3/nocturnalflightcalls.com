import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
import pandas as pd
from pathlib import Path
import random

freq_limits = pd.read_csv("freq_limits.csv")
tables_to_filenames = pd.read_csv("annotation_audio_pairs.csv", index_col='annotation_file').to_dict()['audio_file']

# Spectrogram creation parameters
#sample_rate=22050
window_samples = 256
window_type = 'blackman'
decibel_limits = (-80, 0)

# Denoising parameters
#denoise=True
#quantile=0.8
bandpass = True
order = 4

# How many spectrograms to create and where to save them
max_num_examples = 100
create_audio = True
save_audio = True
save_spectrogram = True

def freq_reformatter(tick_val_hertz, pos):
    """
    Input: float value in Hertz, e.g. 10000.0
    Output: formatted string in kHz, e.g. '10'
    """
    val_khz = tick_val_hertz/1000
    str_format = "%.0f" % val_khz
    if len(str_format) < 2:
        str_format = str_format + '  '
    return str_format


def sec_reformatter(tick_val_sec, pos):
    """
    Input: float value in seconds, e.g. 0.15
    Output: formatted string in milliseconds, e.g. '150'
    """
    val_ms = tick_val_sec*1000
    return "%.0f" % val_ms
    

def save_spectrogram(s, filename, duration, save=True):
    """Create and save an NFC spectrogram
    
    Inputs:
        s: opensoundscape.spectrogram.Spectrogram object
        filename: where to save the file
        duration: duration of the spectrogram
        save: whether or not to save the file
    """
    plt.subplots(figsize=(duration*30, 5))
    plot = s.plot(inline=False)
    s.plot(inline=False)
    ax = plt.gca()
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="y", direction="in", pad=-25, length=5, labelsize=15)
    ax.tick_params(axis="x", direction="in", pad=-20, length=5, labelsize=15)
    
    ax.set_xticks(np.arange(0, duration, 0.05)[1:])
    ax.set_yticks([2000, 4000, 6000, 8000, 10000])
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(sec_reformatter))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(freq_reformatter))
    if save:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close()

num_examples_per_sp = {}


# Loop through all the files
random.seed(42)
keys = list(tables_to_filenames.keys()) 
random.shuffle(keys) # Randomize which files we view

for txt_file in keys:
    audio_file = tables_to_filenames[txt_file]    
    # Loop through all annotations in the file
    df = pd.read_csv(txt_file, sep='\t')
    for idx, row in df.iterrows():
        
        # Decide whether to process this call
        alpha = row['Alpha code']
        if alpha == '?':
            continue
        if alpha not in num_examples_per_sp.keys():
            num_examples_per_sp[alpha] = 1
        elif num_examples_per_sp[alpha] >= max_num_examples:
            continue
        else:
            num_examples_per_sp[alpha] += 1
        
        # Create the filename containing the info about the call
        bp_low, bp_high, approx_duration = freq_limits.query("code == @alpha")[['low_freq', 'high_freq', 'duration']].values[0]
        file_info = [str(x) for x in row[['Order', 'Family', 'Genus', 'Species', 'Alpha code', 'Begin time (s)', 'End time (s)']].tolist()]
        dirname = Path(alpha)
        dirname.mkdir(exist_ok=True)
        filename = '_'.join([Path(txt_file).stem, *file_info])
        print(filename)
        
        begin = row['Begin time (s)']
        end = row['End time (s)']
        
        # Create a longer audio segment to save
        if create_audio:
            audio_dirname = dirname.joinpath('audio')
            audio_dirname.mkdir(exist_ok=True)
            offset = begin + 3.1
            long_duration = 6
            long_segment = Audio.from_file(audio_file, offset=offset, duration=long_duration)
            if save_audio:
                audio_filename = str(audio_dirname.joinpath(filename)) + '.wav'
                long_segment.save(audio_filename)
        
        # Create a shorter spectrogram to show on website
        center = (end + begin)/2
        buffer_s = approx_duration/1.4
        offset = center - approx_duration/2 - buffer_s
        duration = approx_duration + buffer_s*2
        overlap_samples = int(window_samples*0.9)
        
        short_segment = Audio.from_file(audio_file, offset=offset, duration=duration)
        if bandpass:
            short_segment = short_segment.bandpass(bp_low, bp_high, order=order)
        
        s = Spectrogram.from_audio(
            short_segment, window_samples=window_samples, overlap_samples=overlap_samples, decibel_limits=decibel_limits, window_type=window_type)
        spectrogram_dirname = dirname.joinpath('spectrograms')
        spectrogram_dirname.mkdir(exist_ok=True)
        spectrogram_filename = str(spectrogram_dirname.joinpath(filename)) + '.jpg'
        save_spectrogram(s, filename=str(spectrogram_filename), duration=duration, save=save_spectrogram)

    