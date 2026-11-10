# Functions copied from Olsen et al. (2022), public repository:
# https://github.com/MADSOLSEN/SleepStagePrediction
#   signal_processing/normalization.py : downsample, butter_bandpass, butter_bandpass_filter,
#                                        iqr_normalization, iqr_normalization_adaptive, clip_by_iqr
#   signal_processing/spectrogram.py   : cal_psd
# Only change: butter_bandpass_filter uses filtfilt (zero-phase, as in Tallal et al. 2026)
# instead of lfilter. Plotting code was removed.
import sys
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import spectrogram


def downsample(x, fs, fs_out, plot=False):
    if int(fs) is not int(fs_out):
        x_out = signal.resample_poly(x, up=1, down=fs // fs_out)
        return (x_out, fs_out)
    else:
        return (x, fs_out)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(x, fs, lowcut, highcut, order=5, plot=False):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, x)   # CHANGED: original used signal.lfilter
    return y


def iqr_normalization(x, fs, iqr_upper=0.75, iqr_lower=0.25, plot=False):

    def normalize(x):
        x_ = x + np.random.normal(loc=0, scale=sys.float_info.epsilon, size=(x.shape))

        x_iqr_up = np.quantile(x_, iqr_upper)
        x_iqr_lo = np.quantile(x_, iqr_lower)

        return (x_ - x_iqr_lo) / (x_iqr_up - x_iqr_lo + sys.float_info.epsilon) * 2 - 1

    x_norm = np.zeros((x.shape))
    if len(x.shape) == 1:
        x_norm[:] = normalize(x)
    else:
        for n in range(x.shape[1]):
            x_norm[:, n] = normalize(x[:, n])
    return x_norm


def iqr_normalization_adaptive(x, fs, median_window, iqr_window):
    def normalize(x, fs, median_window, iqr_window, iqr_upper=0.75, iqr_lower=0.25):

        # add noise
        x_ = x + np.random.normal(loc=0, scale=sys.float_info.epsilon, size=(x.shape))

        # fix window parameters to odd number
        med_window = (fs * median_window + 1) if (fs * median_window) % 2 == 0 else (fs * median_window)
        iqr_window = (fs * iqr_window + 1) if (fs * iqr_window) % 2 == 0 else (fs * iqr_window)

        # preallocation
        x_med = np.ones((x.shape)) * np.median(x_)
        x_iqr_up = np.ones((x.shape)) * np.quantile(x_, iqr_upper)
        x_iqr_lo = np.ones((x.shape)) * np.quantile(x_, iqr_lower)

        # find rolling median
        x_pd = pd.Series(x_)
        med_ = x_pd.rolling(med_window).median()
        x_med[int(med_window / 2):-int(med_window / 2)] = med_[med_window - 1:]
        x_med[np.isnan(x_med)] = 0  # remove nan

        # find rolling quantiles
        x_iqr_upper = x_pd.rolling(iqr_window).quantile(iqr_upper)
        x_iqr_lower = x_pd.rolling(iqr_window).quantile(iqr_lower)

        # border padding
        x_iqr_up[int(iqr_window / 2):-int(iqr_window / 2)] = x_iqr_upper[iqr_window - 1:]
        x_iqr_lo[int(iqr_window / 2):-int(iqr_window / 2)] = x_iqr_lower[iqr_window - 1:]

        # remove nan
        x_iqr_up[np.isnan(x_iqr_up)] = 0
        x_iqr_lo[np.isnan(x_iqr_lo)] = 0

        # return normalize
        return (x_ - x_iqr_lo) / (x_iqr_up - x_iqr_lo + sys.float_info.epsilon) * 2 - 1

    x_norm = np.zeros((x.shape))
    if len(x.shape) == 1:
        x_norm[:] = normalize(x, fs, median_window, iqr_window)
    else:
        for n in range(x.shape[1]):
            x_norm[:, n] = normalize(x[:, n], fs, median_window, iqr_window)
    return x_norm


def clip_by_iqr(x, fs, threshold=20):

    x[x > threshold] = threshold
    x[x < -threshold] = - threshold

    return x


def cal_psd(x, fs, window, noverlap, nfft, f_min, f_max, f_sub=1, plot_flag=False):
    from scipy.ndimage import maximum_filter as maxfilt
    # border edit
    x_ = np.zeros((x.size + window,))
    x_[window // 2: -window // 2] = x
    x_ = x_ + np.random.normal(loc=0, scale=sys.float_info.epsilon, size=(x_.shape))

    f, t, S = spectrogram(x=x_,
                          fs=fs,
                          window=np.blackman(window),
                          nperseg=window,
                          noverlap=noverlap,
                          nfft=nfft)

    S = S[(f > f_min) & (f <= f_max), :]
    S = maxfilt(np.abs(S), size=(f_sub, 1))
    S = S[::f_sub, :]
    S = np.swapaxes(S, axis1=1, axis2=0)
    S = np.log(S + sys.float_info.epsilon)

    assert (np.sum(np.isinf(S)) == 0)
    assert (np.sum(np.isnan(S)) == 0)
    return S
