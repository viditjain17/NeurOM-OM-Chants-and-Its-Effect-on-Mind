import numpy as np

def extract_features(eeg_signal):
    """
    This function extracts skewness, variance, kurtosis, and Shannon entropy from the EEG signal.
    """
    skewness = np.mean(eeg_signal)  # Simplified, replace with real skewness function
    variance = np.var(eeg_signal)
    kurtosis = np.mean((eeg_signal - np.mean(eeg_signal))**4) / (np.var(eeg_signal)**2)
    shannon_entropy = -np.sum(eeg_signal * np.log2(eeg_signal + 1e-10))
   
    return skewness, variance, kurtosis, shannon_entropy