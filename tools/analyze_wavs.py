import soundfile as sf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import sys

def analyze(path, out_prefix):
    x, sr = sf.read(path)
    x_mono = x.mean(axis=1) if x.ndim>1 else x
    peak = float(np.max(np.abs(x_mono)))
    rms = float(np.sqrt(np.mean(x_mono**2)))
    print(path, 'sr=', sr, 'shape=', x.shape, 'peak=', peak, 'rms=', rms)
    # spectrogram
    f, t, Sxx = signal.spectrogram(x_mono, sr, nperseg=1024)
    plt.figure(figsize=(8,4))
    plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-12), shading='gouraud')
    plt.ylabel('Freq [Hz]'); plt.xlabel('Time [sec]'); plt.title(path)
    plt.colorbar(label='dB')
    plt.tight_layout()
    plt.savefig(out_prefix + '_spec.png')
    plt.close()
    # waveform
    plt.figure(figsize=(8,2))
    times = np.arange(x_mono.size)/sr
    plt.plot(times, x_mono)
    plt.xlabel('Time [s]'); plt.ylabel('Amplitude'); plt.title(path)
    plt.tight_layout()
    plt.savefig(out_prefix + '_wave.png')
    plt.close()

if __name__ == '__main__':
    analyze('audiosmpls/local.wav','audiosmpls/local')
    analyze('audiosmpls/colab.wav','audiosmpls/colab')
    print('done')
