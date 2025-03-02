#!/usr/bin/env python3

"""
pca_extract_no_parselmouth.py

Extracts CSV files from WAV audio using:
  1) PCA-reduced Mel-spectrogram columns (pca_1..pca_N).
  2) Approximate pitch, formant, jitter, shimmer, HNR features (no parselmouth).
  3) Mean/variance of spectral features (MFCC, etc.) in aggregator form.

Optional: call Deepgram to get transcriptions (comment out if not needed).

Usage:
  python pca_extract_no_parselmouth.py [inputdir]

  By default, it processes .wav files under 'clips' directory if no other input
  directory is specified.

Outputs:
  For each .wav file, a matching .csv file is created that holds frame-level data.
  If Deepgram is invoked, a .json transcription is also produced.
"""

import os
import sys
import json

import wave
import librosa
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import httpx

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)

###############################################################################
# 1. Optional Transcription (Deepgram)
###############################################################################
def get_transcription_json(wav_path):
    """
    get_transcription_json(wav_path)
    --------------------------------
    Checks if a JSON transcription for the given WAV file already exists.
    If not, it calls Deepgram to create one. The output is saved as a .json
    file with the same basename as the .wav.

    Parameters
    ----------
    wav_path : str
        Path to the input WAV file.

    Returns
    -------
    None
    """
    # Create JSON filename based on the .wav filename
    json_filename = wav_path.replace(".wav", ".json")

    # If the JSON file already exists, do nothing
    if os.path.exists(json_filename):
        return

    # Initialize the DeepgramClient with default settings
    deepgram: DeepgramClient = DeepgramClient()

    # Read the WAV file into memory
    with open(wav_path, "rb") as file:
        buffer_data = file.read()

    # Create the payload with the in-memory audio
    payload: FileSource = {
        "buffer": buffer_data,
    }

    # Set transcription options as needed
    options: PrerecordedOptions = PrerecordedOptions(
        model="nova-2",
    )

    # Call the Deepgram API with a generous timeout in case of large files
    response = deepgram.listen.rest.v("1").transcribe_file(
        payload, options, timeout=httpx.Timeout(300.0, connect=10.0)
    )

    # Save response to a JSON file
    with open(json_filename, "w") as f:
        f.write(response.to_json(indent=4))


###############################################################################
# 2. Approximate Jitter, Shimmer, HNR
###############################################################################
def calc_jitter_shimmer_hnr(audio, sr, pitch_frames, frame_hop):
    """
    calc_jitter_shimmer_hnr(audio, sr, pitch_frames, frame_hop)
    -----------------------------------------------------------
    Approximates jitter, shimmer, and HNR (harmonic-to-noise ratio)
    from time-domain audio without using parselmouth/Praat. This is
    a naive approach intended as a rough stand-in.

    Parameters
    ----------
    audio : np.array
        Time-domain signal.
    sr : int
        Sampling rate in Hz.
    pitch_frames : np.array
        Array of pitch values (f0) in Hz, one pitch estimate per frame.
    frame_hop : int
        Hop length in samples between frames.

    Returns
    -------
    jitter : float
        Relative average perturbation in the pitch period.
    shimmer : float
        Relative average perturbation in amplitude.
    hnr : float
        A rough harmonic-to-noise ratio in dB.
    """

    # Convert pitch to period in samples (sr / f0)
    period_samples = []
    for f0 in pitch_frames:
        if f0 <= 0.0:
            period_samples.append(0.0)
        else:
            period_samples.append(sr / f0)

    # Filter out unvoiced frames (where pitch = 0)
    voiced_periods = np.array([p for p in period_samples if p > 0])
    if len(voiced_periods) < 2:
        # If not enough voiced frames, return zeros
        return (0.0, 0.0, 0.0)

    # Jitter: relative average absolute difference of consecutive pitch periods
    period_diffs = np.abs(np.diff(voiced_periods))
    avg_period_diff = np.mean(period_diffs)
    avg_period = np.mean(voiced_periods)
    jitter = float(avg_period_diff / (avg_period + 1e-8))

    # Shimmer: uses RMS amplitude differences between consecutive voiced frames
    # We'll approximate amplitude as RMS in each frame of length 'frame_hop'
    frame_size = frame_hop
    amplitude_env = []
    for i, f0 in enumerate(pitch_frames):
        start = i * frame_hop
        end = start + frame_size
        if end > len(audio):
            break
        frame = audio[start:end]
        amp = np.sqrt(np.mean(frame**2))
        amplitude_env.append(amp)

    # Keep only amplitude values for frames with pitch > 0
    voiced_amplitudes = np.array([
        amplitude_env[i] for i, f0 in enumerate(pitch_frames)
        if f0 > 0 and i < len(amplitude_env)
    ])
    if len(voiced_amplitudes) < 2:
        return float(jitter), 0.0, 0.0

    amp_diffs = np.abs(np.diff(voiced_amplitudes))
    avg_amp_diff = np.mean(amp_diffs)
    avg_amp = np.mean(voiced_amplitudes)
    shimmer = float(avg_amp_diff / (avg_amp + 1e-8))

    # HNR (rough approach):
    #   ratio of voiced energy to (total_energy - voiced_energy)
    total_energy = np.sum(audio**2)
    if total_energy < 1e-8:
        return float(jitter), float(shimmer), 0.0

    voiced_energy = 0.0
    for i, f0 in enumerate(pitch_frames):
        if f0 > 0:
            start = i * frame_hop
            end = start + frame_size
            if end > len(audio):
                break
            chunk = audio[start:end]
            voiced_energy += np.sum(chunk**2)

    noise_energy = total_energy - voiced_energy
    if noise_energy < 1e-8:
        # Means almost all is voiced
        hnr = 100.0
    else:
        hnr = 10.0 * np.log10((voiced_energy + 1e-8) / (noise_energy + 1e-8))

    return float(jitter), float(shimmer), float(hnr)


###############################################################################
# 3. Approximate Formants via LPC
###############################################################################
def estimate_formants_lpc(audio, sr, frame_size=0.025, hop_size=0.0125, lpc_order=8):
    """
    estimate_formants_lpc(audio, sr, frame_size, hop_size, lpc_order)
    -----------------------------------------------------------------
    Approximates the first 4 formant frequencies using a naive LPC approach.

    Process:
      1) Split audio into frames of ~25ms with 50% overlap.
      2) Apply LPC on each windowed frame => find roots => pick formant frequencies.
      3) Track average (and standard deviation) of the first 4 formants across frames.

    Parameters
    ----------
    audio : np.array
        Time-domain signal.
    sr : int
        Sampling rate in Hz.
    frame_size : float
        Duration of each frame in seconds (default 0.025 => 25ms).
    hop_size : float
        Overlap: typically half of frame_size (default 0.0125 => 12.5ms).
    lpc_order : int
        Order of the LPC analysis.

    Returns
    -------
    (f1_mean, f1_std, f2_mean, f2_std, f3_mean, f3_std, f4_mean, f4_std)
        Each value is a float representing average or std dev of the formant freq.
    """
    frame_len = int(sr * frame_size)
    hop_len = int(sr * hop_size)
    if hop_len < 1:
        hop_len = 1

    all_f1, all_f2, all_f3, all_f4 = [], [], [], []
    idx = 0
    while idx + frame_len <= len(audio):
        frame = audio[idx: idx + frame_len]
        # Window the frame to reduce edge effects
        frame_win = frame * np.hanning(len(frame))

        try:
            # Use librosa's LPC function
            lpc_coeffs = librosa.lpc(frame_win, order=lpc_order)
            # Solve for polynomial roots
            roots = np.roots(lpc_coeffs)
            # Consider only roots in the upper half of the Z-plane
            roots = [r for r in roots if np.imag(r) >= 0]
            angles = np.angle(roots)
            freqs = angles * (sr / (2 * np.pi))
            # Keep only formants < Nyquist
            freqs = [f for f in freqs if 0 < f < (sr / 2)]
            freqs = sorted(freqs)
            # Zero-pad if < 4
            while len(freqs) < 4:
                freqs.append(0.0)
            all_f1.append(freqs[0])
            all_f2.append(freqs[1])
            all_f3.append(freqs[2])
            all_f4.append(freqs[3])
        except:
            # If LPC fails (e.g., silence), store zeros
            all_f1.append(0.0)
            all_f2.append(0.0)
            all_f3.append(0.0)
            all_f4.append(0.0)

        idx += hop_len

    return (
        np.mean(all_f1), np.std(all_f1),
        np.mean(all_f2), np.std(all_f2),
        np.mean(all_f3), np.std(all_f3),
        np.mean(all_f4), np.std(all_f4)
    )


###############################################################################
# 4. Additional Spectral Features (mean/var of MFCC, etc.)
###############################################################################
def compute_spectral_features(audio, sr, frame_size=0.032, hop_pct=0.5, n_mfcc=13):
    """
    compute_spectral_features(audio, sr, frame_size, hop_pct, n_mfcc)
    -----------------------------------------------------------------
    Computes additional aggregated spectral features, specifically:
      - Mean & variance of MFCCs
      - Mean & variance of a "cepstrum" array (also using MFCC call)

    These aggregated features act as global or multi-frame descriptors.

    Parameters
    ----------
    audio : np.array
        Time-domain signal.
    sr : int
        Sampling rate in Hz.
    frame_size : float
        Window size in seconds for STFT (default ~32ms).
    hop_pct : float
        Percentage overlap (default 0.5 => 50% overlap).
    n_mfcc : int
        Number of MFCC coefficients to compute (default 13).

    Returns
    -------
    out : dict
        Dictionary of aggregated feature statistics like 'mfcc_mean_1',
        'mfcc_var_1', 'cep_mean_1', 'cep_var_1', etc.
    """
    frame_length = int(sr * frame_size)
    hop_length = int(frame_length * hop_pct)
    if hop_length < 1:
        hop_length = 1

    # Compute MFCC
    mfcc_feats = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc,
        n_fft=frame_length, hop_length=hop_length
    )
    mfcc_mean = np.mean(mfcc_feats, axis=1)
    mfcc_var = np.var(mfcc_feats, axis=1)

    # Compute a placeholder "cepstrum" also using MFCC pipeline on STFT magnitude
    S = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    S_mag = np.abs(S)
    cepstrum = librosa.feature.mfcc(
        S=librosa.power_to_db(S_mag**2),
        sr=sr,
        n_mfcc=n_mfcc
    )
    cep_mean = np.mean(cepstrum, axis=1)
    cep_var = np.var(cepstrum, axis=1)

    out = {}
    for i in range(n_mfcc):
        out[f"mfcc_mean_{i+1}"] = mfcc_mean[i]
        out[f"mfcc_var_{i+1}"] = mfcc_var[i]
        out[f"cep_mean_{i+1}"] = cep_mean[i]
        out[f"cep_var_{i+1}"] = cep_var[i]

    return out


###############################################################################
# 5. Main PCA Extraction
###############################################################################
def wav_to_pca_csv(
    wav_path,
    sample_rate=48000,
    duration=None,
    n_fft=1024,
    hop_length=256,
    n_mels=20,
    n_components=10,
    fmin=50.0,
    fmax=1000.0
):
    """
    wav_to_pca_csv(...)
    ------------------
    Main function to convert a single .wav file into a .csv with:
      - PCA applied to a Mel-spectrogram
      - Approximate pitch, energy, spectral bandwidth
      - Approximate jitter, shimmer, HNR
      - LPC-based formant estimates
      - Additional aggregator features (MFCC mean/var, etc.)
      - Word-level placeholders if a Deepgram JSON transcription exists

    Parameters
    ----------
    wav_path : str
        Path to the .wav file.
    sample_rate : int
        Desired resampling rate (default 48000).
    duration : float
        If not None, loads only up to this many seconds of audio.
    n_fft : int
        FFT size for the Mel-spectrogram (default 1024).
    hop_length : int
        Hop size for the Mel-spectrogram (default 256).
    n_mels : int
        Number of Mel filter banks (default 20).
    n_components : int
        Number of PCA components for dimensionality reduction (default 10).
    fmin : float
        Minimum frequency for pitch detection and Mel filters (default 50 Hz).
    fmax : float
        Maximum frequency for pitch detection and Mel filters (default 1000 Hz).

    Returns
    -------
    None
        A .csv file is written to disk with the same basename as the .wav file.
    """
    csv_output_path = wav_path.replace(".wav", ".csv")
    if os.path.exists(csv_output_path):
        return

    # 1) Load audio
    audio, sr = librosa.load(wav_path, sr=sample_rate, duration=duration)
    # Preemphasis filter to boost high frequencies
    audio = librosa.effects.preemphasis(audio, coef=0.97)

    # 2) Mel-spectrogram
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=2.0,
        fmin=fmin, fmax=fmax
    )
    S_db = librosa.power_to_db(S, ref=np.max).T  # shape (time_frames, n_mels)

    # 3) PCA on the Mel-spectrogram
    pca = PCA(n_components=n_components)
    S_pca = pca.fit_transform(S_db)
    frame_time_sec = np.arange(S_pca.shape[0]) * (hop_length / sr)

    # 4) Approximate pitch using librosa.pyin
    f0, _, _ = librosa.pyin(
        audio, sr=sr, fmin=fmin, fmax=fmax,
        frame_length=n_fft, hop_length=hop_length
    )
    f0 = np.nan_to_num(f0, nan=0.0)

    # 5) Energy (RMS) and spectral bandwidth
    energy = librosa.feature.rms(
        y=audio,
        frame_length=n_fft,
        hop_length=hop_length
    )[0]
    spec_bw = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]

    # Align the arrays by truncating to the smallest length
    min_len = min(len(S_pca), len(f0), len(energy), len(spec_bw))
    S_pca = S_pca[:min_len]
    frame_time_sec = frame_time_sec[:min_len]
    f0 = f0[:min_len]
    energy = energy[:min_len]
    spec_bw = spec_bw[:min_len]

    # Build a DataFrame for frame-wise data
    pca_cols = [f"pca_{i+1}" for i in range(n_components)]
    data = np.column_stack((frame_time_sec, S_pca))
    columns = ["time_sec"] + pca_cols
    df = pd.DataFrame(data, columns=columns)

    df["pitch_hz"] = f0
    df["energy"] = energy
    df["spec_bw"] = spec_bw

    # 6) Approx. jitter, shimmer, HNR (stored in every frame for convenience)
    jitter, shimmer, hnr = calc_jitter_shimmer_hnr(
        audio, sr, pitch_frames=f0, frame_hop=hop_length
    )
    df["jitter"] = jitter
    df["shimmer"] = shimmer
    df["hnr"] = hnr

    # 7) Approx formants via LPC
    (
        f1_mean, f1_std, f2_mean, f2_std,
        f3_mean, f3_std, f4_mean, f4_std
    ) = estimate_formants_lpc(audio, sr)
    df["f1_mean"] = f1_mean
    df["f1_std"] = f1_std
    df["f2_mean"] = f2_mean
    df["f2_std"] = f2_std
    df["f3_mean"] = f3_mean
    df["f3_std"] = f3_std
    df["f4_mean"] = f4_mean
    df["f4_std"] = f4_std

    # 8) Aggregated spectral features
    spec_agg = compute_spectral_features(audio, sr)
    for k, v in spec_agg.items():
        df[k] = v

    # 9) Word-level placeholders
    df["word"] = "-"
    df["word_seq"] = "-"
    df["word_dur"] = 0.0

    # 10) If JSON transcription (Deepgram) is available, parse and add alignment
    json_path = wav_path.replace(".wav", ".json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            transcription_data = json.load(f)

        # Deepgram returns words under this path
        words_info = transcription_data["results"]["channels"][0]["alternatives"][0]["words"]
        for i in range(len(df)):
            tsec = df.at[i, "time_sec"]
            matched_word = "-"
            matched_dur = 0.0
            for w in words_info:
                start, end = w["start"], w["end"]
                if start <= tsec < end:
                    matched_word = w["word"]
                    matched_dur = end - start
                    break
            df.at[i, "word"] = matched_word
            df.at[i, "word_dur"] = matched_dur

        # Mark "START" vs. "CONTINUE" of each word
        for i in range(len(df)):
            if i == 0 or df.at[i, "word"] != df.at[i - 1, "word"]:
                if df.at[i, "word"] != "-":
                    df.at[i, "word_seq"] = "START"
            else:
                if df.at[i, "word"] != "-":
                    df.at[i, "word_seq"] = "CONTINUE"

    # Write the DataFrame to CSV
    df.to_csv(csv_output_path, index=False)
    print(f"[PCA-Extraction] CSV saved at {csv_output_path}")


def process_all_wav(inputdir):
    """
    process_all_wav(inputdir)
    -------------------------
    Recursively scans 'inputdir' for .wav files, optionally runs transcription,
    and calls 'wav_to_pca_csv' to create .csv for each .wav.

    Parameters
    ----------
    inputdir : str
        Directory containing .wav files (possibly in subfolders).

    Returns
    -------
    None
    """
    for root, dirs, files in os.walk(inputdir):
        for filename in files:
            if filename.endswith(".wav"):
                full_path = os.path.join(root, filename)
                print(f"Processing: {full_path}")
                # If needed, call Deepgram to get JSON transcription
                get_transcription_json(full_path)  # remove or comment out if not needed
                wav_to_pca_csv(full_path)


if __name__ == "__main__":
    """
    Entry point: can specify the input directory as a command-line argument or
    default to 'clips'.
    """
    inputdir = "clips"
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        inputdir = sys.argv[1]
    process_all_wav(inputdir)
