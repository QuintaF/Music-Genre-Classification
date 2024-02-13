'''
In this file are implemented all the functions 
for the extraction of the audio features used in the 
classification process. 
Most of the functions can provide a graph.
'''

#directory change ->  ...\Music-Genre-Classification\main\src
import os, sys
file_path = os.path.dirname(__file__)
os.chdir(file_path)

#modules
import numpy as np
import matplotlib.pyplot as plt
import librosa
import time

#global
FRAME_LENGTH = 2048
HOP_LENGTH = 512


def get_filenames():
    '''
    retrieves filenames from the folder

    :returns: list of filenames
    '''
    filenames = []
    # open audio files directory
    path = "../dataset/Data/genres_original/"
    for genre in os.listdir(path):
        for song in os.listdir(path + genre):
            filenames.append(path + genre + '/' + song)

    return filenames


def mel_spectrogram(song, data, sr):
    '''
    computes the mel spectrogram

    :saves image: at dataset/My_Data/mel_spectrogram/name.jpg
    '''

    name = song.rsplit('/',1)[1]
    plt.figure(name,figsize=(17,5))
    mel = librosa.feature.melspectrogram(y = data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH ,dtype="float64")
    mel_db = librosa.power_to_db(mel, ref=np.max)

    librosa.display.specshow(mel_db, y_axis="mel", x_axis="time")
    plt.colorbar(format="%+2.0f dB")
    plt.set_cmap("magma")
    #plt.clim(-40,0)
    plt.savefig("../dataset/My_Data/mel_spectrograms/" + name[:-3] + "jpg")
    plt.close()

def chroma(song, data, sr, plot = False, full = False):
    '''
    computes chroma feature

    :returns: mean, var
    '''

    # compute
    chroma = librosa.feature.chroma_stft(y = data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH, dtype="float64")
    mean = np.mean(chroma)
    var = np.var(chroma)

    if plot:
        # chromagram
        plt.figure(song.rsplit('/',1)[1],figsize=(17,5))
        librosa.display.specshow(chroma, x_axis="time", y_axis="chroma", cmap="magma")
        plt.colorbar()
        plt.show()

    elif full:
        return chroma
    
    return mean, var


def root_mean_square_energy(song, data, sr, plot = False, full = False):
    '''
    computes root mean square energy

    :returns: mean, var
    '''

    rms = librosa.feature.rms(y=data, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mean = np.mean(rms)
    var = np.var(rms)

    if plot:
        # signal plot
        plt.figure(song.rsplit('/',1)[1],figsize=(17,5))
        librosa.display.waveshow(data, label="signal", alpha=0.5, color="#1f77b4") 

        # rms plot
        t = np.linspace(0, len(data), num = len(rms[0]))/sr
        plt.plot(t, rms[0], color = 'r')
        plt.show()

    elif full:
        return rms
    
    return mean, var


def spectral_centroid_and_bandwidth(song, data, sr, plot = False, full = False):
    '''
    computes spectral centroid and bandwidth

    :returns: mean, var
    '''

    # compute centroid
    centroid = librosa.feature.spectral_centroid(y = data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    c_mean = np.mean(centroid)
    c_var = np.var(centroid)

    # compute bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y = data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    b_mean = np.mean(bandwidth)
    b_var = np.var(bandwidth)

    if plot:
        # spectrogram plot
        plt.figure(song.rsplit('/',1)[1],figsize=(17,5))
        stft = librosa.stft(data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        db_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        librosa.display.specshow(db_stft, x_axis="time", y_axis="log", sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        plt.colorbar(format='%+2.0f dB')
        
        # centroid/bandwidth plot 
        t = np.linspace(0, len(centroid[0]) - 1, num = len(centroid[0])) * HOP_LENGTH/sr
        plt.plot(t, centroid[0].T, color = "white", label="spectral centroid")
        plt.plot(t, bandwidth[0].T, color = "black", label="spectral bw")
        plt.fill_between(t, np.maximum(0, centroid[0] - bandwidth[0]), np.minimum(centroid[0] + bandwidth[0], sr/2), alpha=0.5, label='centroid +- bandwidth', color="white")
        plt.legend(loc="upper right")
        plt.show()

    elif full:
        return centroid, bandwidth
    
    return c_mean, c_var, b_mean, b_var


def spectral_rolloff(song, data, sr = None, plot = False, full = False):
    '''
    computes the spectral rolloff: the frequency below which a
    percentage of the frequencies lie

    :returns: list of (rolloff_mean, rolloff_var) for all files
    '''

    # compute
    rolloff = librosa.feature.spectral_rolloff(y = data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mean = np.mean(rolloff)
    var = np.var(rolloff)

    if plot:
        # spectrogram plot
        plt.figure(song.rsplit('/',1)[1],figsize=(17,5))
        stft = librosa.stft(data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        db_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        librosa.display.specshow(db_stft, x_axis="time", y_axis="log", sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        plt.colorbar(format='%+2.0f dB')
        
        # rolloff plot 
        t = np.linspace(0, len(rolloff[0]) - 1, num = len(rolloff[0])) * HOP_LENGTH/sr
        plt.plot(t, rolloff[0].T, color = "white", label="spectral rolloff")
        plt.legend(loc="upper right")
        plt.show()

    elif full:
        return rolloff

    return mean, var


def zero_crossing_rate(song, data, sr = None, plot = False, full = False):
    '''
    computes the zero crossing rate for the files

    :returns: list of (zcr_mean, zcr_var) for all files
    '''

    # compute
    zcr = librosa.feature.zero_crossing_rate(y = data, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mean = np.mean(zcr)
    var = np.var(zcr)

    if plot:
        # zcr plot
        plt.figure(song.rsplit('/',1)[1],figsize=(17,5))

        t_zcr = np.linspace(0, len(data), num = len(zcr[0]))/sr
        plt.plot(t_zcr,zcr[0])

        plt.xlabel("time")
        plt.title("Zero Crossing Rate")
        plt.legend()
        plt.show()

    elif full:
        return zcr
    
    return mean, var


def decompose_harmonic_percussive(song, data, sr, plot = False, full = False):
    '''
    computes harmonic values

    :returns: mean, var
    '''

    # compute
    harmonic, percussive = librosa.effects.hpss(y = data)
    h_mean = np.mean(harmonic)
    h_var = np.var(harmonic)
    p_mean = np.mean(percussive)
    p_var = np.var(percussive)

    if plot:
        # harmonic/percussive plot
        plt.figure(song.rsplit('/',1)[1],figsize=(17,5))
        
        librosa.display.waveshow(harmonic, label="harmonic", alpha=.5, color="#1f77b4")
        librosa.display.waveshow(percussive, label="percussive", alpha=.5, color="#d62728")
        plt.legend(loc = "upper left")
        plt.show()

        # harmonic/percussive spectrogram
        plt.figure(song.rsplit('/',1)[1],figsize=(17,5))
        f = librosa.stft(data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        harmonic, percussive  = librosa.decompose.hpss(f) 
        plt.subplot(3,1,1)
        plt.title("Harmonic")
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(harmonic), ref=np.max), y_axis="log", x_axis="time")
        plt.subplot(3,1,3)
        plt.title("Percussive")
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(percussive), ref=np.max), y_axis="log", x_axis="time")
        
        plt.show()
    
    elif full:
        return harmonic, percussive

    return h_mean, h_var, p_mean, p_var


def tempo(song, data, sr, plot = False, full = False):
    '''
    computes tempo as beat per minute(bpm)

    :returns: bpm
    '''

    # compute
    bpm = librosa.feature.tempo(y = data, sr=sr, hop_length=HOP_LENGTH)

    if plot:
        onset_env = librosa.onset.onset_strength(y=data, sr=sr)
        dtempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr,aggregate=None)

        # tempo estimation plot
        plt.figure(song.rsplit('/',1)[1],figsize=(17,5))
        tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        librosa.display.specshow(tg, x_axis='time', y_axis='tempo', cmap='magma')
        plt.plot(librosa.times_like(dtempo), dtempo, color='c', linewidth=1.5, label='Tempo estimate (default prior)')
        plt.show()
    elif full:
        return bpm

    return bpm[0]


def band_energy_ratio(song, data, sr, split_freq = 2000, plot = False, full = False):
    '''
    computes band energy ratio for different frames

    :returns: mean, var
    '''

    # compute 
    stft = librosa.stft(y=data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)  #spectrogram
    max_freq = sr/2  #max sampled frequency
    num_bins = stft.shape[0]
    freq_bin_distance = max_freq/num_bins

    split_freq_bin = int(np.floor(split_freq/freq_bin_distance))  #round down
    stft = np.abs(stft, dtype=np.float64)
    power_spectrogram = (stft ** 2).T  #(time, pwr)

    bers = []
    # ber for each frame
    for freq in power_spectrogram:
        low_band_sum = np.sum(freq[:split_freq_bin])
        high_band_sum = np.sum(freq[split_freq_bin:])
        
        if high_band_sum == 0:
            high_band_sum = 1e-12

        ber = low_band_sum/ high_band_sum
        bers.append(ber)

    bers = np.array(bers)
    mean = np.mean(bers)
    var = np.var(bers)
                  
    if plot:
        # ber plot
        plt.figure(song.rsplit('/',1)[1],figsize=(17,5))
        t = librosa.times_like(bers, hop_length=HOP_LENGTH, n_fft=FRAME_LENGTH)
        plt.plot(t, bers)
        plt.show()

    elif full:
        return bers

    return mean, var


def amplitude_envelope(song, data, sr, plot = False, full = False):
    '''
    computes amplitude envelope for different frames

    :returns: mean, var
    '''

    # compute for each frame
    amp_envs = []
    for i in range(0, len(data), HOP_LENGTH):
        frame = data[i:i+FRAME_LENGTH]  #get frame
        amp_envs.append(max(frame))  #max amplitude in the current frame

    amp_envs = np.array(amp_envs)
    mean = np.mean(amp_envs)
    var = np.var(amp_envs)
                  
    if plot:
        plt.figure(song.rsplit('/',1)[1],figsize=(17,5))

        # signal plot
        librosa.display.waveshow(data, label="signal", alpha=.5, color="#1f77b4") 

        # ae plot
        t = np.linspace(0, len(amp_envs), num=len(amp_envs)) * HOP_LENGTH/sr
        plt.plot(t, amp_envs, color='r')
        plt.xlabel = "time"
        plt.xlabel = "time"
        plt.show()
    
    elif full:
        return amp_envs

    return mean, var


def mel_frequency_cepstral_coefficients(song, data, sr, full = False):
    '''
    computes the first 20 mfccs 

    :returns: mean, var
    '''

    # compute
    mfccs = librosa.feature.mfcc(y=data, n_mfcc=20)
    means = np.mean(mfccs, axis=1)
    var = np.var(mfccs, axis=1)

    if full:
        return mfccs

    return means, var


def extraction_pipeline():

    filenames = get_filenames()

    all_features = []
    start = time.time()
    for song in filenames[0:]:
        song_features = np.empty(61) # features
        data, sr = librosa.load(song)

        #mel_spectrogram(song, data, sr)
        song_features[0], song_features[1] = chroma(song, data, sr)
        song_features[2], song_features[3] = root_mean_square_energy(song, data, sr)
        song_features[4], song_features[5], song_features[6], song_features[7] = spectral_centroid_and_bandwidth(song, data, sr)
        song_features[8], song_features[9] = spectral_rolloff(song, data, sr)
        song_features[10], song_features[11] = zero_crossing_rate(song, data ,sr)
        song_features[12], song_features[13], song_features[14], song_features[15] = decompose_harmonic_percussive(song, data ,sr)
        song_features[16] = tempo(song, data, sr)
        song_features[17], song_features[18] = band_energy_ratio(song, data, sr)
        song_features[19], song_features[20] = amplitude_envelope(song, data, sr)
        song_features[21:61:2], song_features[22:62:2] = mel_frequency_cepstral_coefficients(song, data, sr)

        all_features.append(song_features)

    end = time.time()
    print(end - start)
    # EXECUTION TIME: 2416 sec
    # Mel spectrogram generation: 737 sec
    
    np.save("../dataset/My_Data/features.npy", np.array(all_features))
    
    return 0


if __name__ == '__main__':
    extraction_pipeline()