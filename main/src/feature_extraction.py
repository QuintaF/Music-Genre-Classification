#directory change ->  ...\Music-Genre-Classification\main\src
import os, sys
file_path = os.path.dirname(__file__)
os.chdir(file_path)

#modules
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
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
    plt.show()


def chroma(song, data, sr, plot = False, full = False):
    '''
    computes chroma feature

    :returns: mean, std
    '''

    # compute
    chroma = librosa.feature.chroma_stft(y = data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH, dtype="float64")
    mean = np.mean(chroma)
    std = np.std(chroma)

    if plot:
        # chromagram
        plt.figure(song.rsplit('/',1)[1],figsize=(17,5))
        librosa.display.specshow(chroma, x_axis="time", y_axis="chroma", cmap="magma")
        plt.colorbar()
        plt.show()

    elif full:
        return chroma
    
    return mean, std


def root_mean_square_energy(song, data, sr, plot = False, full = False):
    '''
    computes root mean square energy

    :returns: mean, std
    '''

    rms = librosa.feature.rms(y=data, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mean = np.mean(rms)
    std = np.std(rms)

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
    
    return mean, std


def spectral_centroid_and_bandwidth(song, data, sr, plot = False, full = False):
    '''
    computes spectral centroid and bandwidth

    :returns: mean, std
    '''

    # compute centroid
    centroid = librosa.feature.spectral_centroid(y = data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    c_mean = np.mean(centroid)
    c_std = np.std(centroid)

    # compute bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y = data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    b_mean = np.mean(bandwidth)
    b_std = np.std(bandwidth)

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
        plt.fill_between(t, np.maximum(0, centroid[0] - bandwidth[0]), np.minimum(centroid[0] + bandwidth[0], sr/2), alpha=0.5, label='centroid +- bandwidth', color="white")
        plt.legend(loc="upper right")
        plt.show()

    elif full:
        return centroid, bandwidth
    
    return c_mean, c_std, b_mean, b_std


def spectral_rolloff(song, data, sr = None, plot = False, full = False):
    '''
    computes the spectral rolloff: the frequency below which a
    percentage of the frequencies lie

    :returns: list of (rolloff_mean, rolloff_std) for all files
    '''

    # compute
    rolloff = librosa.feature.spectral_rolloff(y = data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mean = np.mean(rolloff)
    std = np.std(rolloff)

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

    return mean, std


def zero_crossing_rate(song, data, sr = None, plot = False, full = False):
    '''
    computes the zero crossing rate for the files

    :returns: list of (zcr_mean, zcr_std) for all files
    '''

    # compute
    zcr = librosa.feature.zero_crossing_rate(y = data, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mean = np.mean(zcr)
    std = np.std(zcr)

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
    
    return mean, std


def decompose_harmonic_percussive(song, data, sr, plot = False, full = False):
    '''
    computes harmonic values

    :returns: mean, std
    '''

    # compute
    harmonic, percussive = librosa.effects.hpss(y = data)
    h_mean = np.mean(harmonic)
    h_std = np.std(harmonic)
    p_mean = np.mean(percussive)
    p_std = np.std(percussive)

    if plot:
        # harmonic/percussive plot
        plt.figure(song.rsplit('/',1)[1],figsize=(17,5))
        
        librosa.display.waveshow(harmonic, label="harmonic", alpha=.5, color="#1f77b4")
        librosa.display.waveshow(percussive, label="percussive", alpha=.5, color="#d62728")
        plt.legend(loc = "upper left")

        # harmonic/percussive spectrogram
        '''
        f = librosa.stft(data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        harmonic, percussive  = librosa.decompose.hpss(f) 
        plt.subplot(3,1,2)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(harmonic), ref=np.max), y_axis="log", x_axis="time")
        plt.subplot(3,1,3)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(percussive), ref=np.max), y_axis="log", x_axis="time")
        '''
        plt.show()
    
    elif full:
        return harmonic, percussive

    return h_mean, h_std, p_mean, p_std


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

    :returns: mean, std
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
    std = np.std(bers)
                  
    if plot:
        # ber plot
        plt.figure(song.rsplit('/',1)[1],figsize=(17,5))
        t = librosa.times_like(bers, hop_length=HOP_LENGTH, n_fft=FRAME_LENGTH)
        plt.plot(t, bers)
        plt.show()

    elif full:
        return bers

    return mean, std


def amplitude_envelope(song, data, sr, plot = False, full = False):
    '''
    computes amplitude envelope for different frames

    :returns: mean, std
    '''

    # compute for each frame
    amp_envs = []
    for i in range(0, len(data), HOP_LENGTH):
        frame = data[i:i+FRAME_LENGTH]  #get frame
        amp_envs.append(max(frame))  #max amplitude in the current frame

    amp_envs = np.array(amp_envs)
    mean = np.mean(amp_envs)
    std = np.std(amp_envs)
                  
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

    return mean, std


def mel_frequency_cepstral_coefficients(song, data, sr, full = False):
    '''
    computes the first 20 mfccs 

    :returns: mean, std
    '''

    # compute
    mfccs = librosa.feature.mfcc(y=data, n_mfcc=20)
    means = np.mean(mfccs, axis=1)
    stds = np.std(mfccs, axis=1)

    if full:
        return mfccs

    return means, stds


def extraction_pipeline():

    labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    filenames = get_filenames()

    all_features = []
    start = time.time()
    for idx, song in enumerate(filenames):
        song_features = np.zeros(62, dtype=object) # features + label
        data, sr = librosa.load(song)

        '''
        graph of ft and spectrogram with rolloff
        plt.figure()
        
        fft = np.fft.fft(data)
        fft = fft[0:len(fft)//2]
        fft[1::] = 2*fft[1::]
        f = np.linspace(0,len(fft)//2 -1, num = len(fft))*(sr/len(data)) 
        plt.plot(f, np.abs(fft))
        
        
        plt.figure()
        da = librosa.stft(data)
        librosa.display.specshow(librosa.power_to_db(np.abs(da), ref=np.max), y_axis="log", x_axis="time")
        # compute
        rolloff = librosa.feature.spectral_rolloff(y = data)
        plt.colorbar()
        plt.set_cmap('magma')
        plt.plot(librosa.times_like(rolloff),rolloff[0], color='green')
        plt.show()
        '''

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
        
        new_idx = idx + (idx//500)  # jazz(at 500) has 1 less file
        song_features[61] = labels[new_idx//100]
        all_features.append(song_features)

    end = time.time()
    # EXECUTION TIME: 3932.1247568130493 sec
    print(end - start)
    df = pd.DataFrame(np.array(all_features))
    df.columns = ["chroma_stft_mean","chroma_stft_std","rms_mean","rms_std","spectral_centroid_mean","spectral_centroid_std",
                                                               "spectral_bandwidth_mean","spectral_bandwidth_std","rolloff_mean","rolloff_std","zero_crossing_rate_mean","zero_crossing_rate_std",
                                                               "harmonic_mean","harmonic_std","percussive_mean","percussive_std","tempo","band_energy_ratio_mean","band_energy_ratio_std",
                                                               "amplitude_envelope_mean","amplitude_envelope_std","mfcc1_mean","mfcc1_std","mfcc2_mean","mfcc2_std","mfcc3_mean","mfcc3_std",
                                                               "mfcc4_mean","mfcc4_std","mfcc5_mean","mfcc5_std","mfcc6_mean","mfcc6_std","mfcc7_mean","mfcc7_std","mfcc8_mean","mfcc8_std", 
                                                               "mfcc9_mean","mfcc9_std","mfcc10_mean","mfcc10_std","mfcc11_mean","mfcc11_std","mfcc12_mean","mfcc12_std","mfcc13_mean","mfcc13_std",
                                                               "mfcc14_mean","mfcc14_std","mfcc15_mean","mfcc15_std","mfcc16_mean","mfcc16_std","mfcc17_mean","mfcc17_std","mfcc18_mean","mfcc18_std",
                                                               "mfcc19_mean","mfcc19_std","mfcc20_mean","mfcc20_std","label"]
    
    df.to_csv("../dataset/My_Data/features.csv", sep=',', header=True, index=False)
    
    return 0


if __name__ == '__main__':
    extraction_pipeline()