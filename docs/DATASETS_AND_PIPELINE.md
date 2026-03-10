# EEG Datasets and Data Pipeline Guide

This doc recommends public EEG datasets for cognitive-state work (focus, relaxation, stress) and outlines how to use them in the Brain State Dashboard.

---

## Recommended datasets

### 1. **EEG Dataset of Relaxation and Concentration Moods** (primary)

- **Where:** [Mendeley Data – 8c26dn6c7w](https://data.mendeley.com/datasets/8c26dn6c7w/1)
- **Why:** Directly targets **relaxation vs concentration**; recorded with **EMOTIV EPOC+** (14 channels, 250 Hz), so it matches your EMOTIV-focused stack.
- **Content:** 30 participants, 4 sessions each. Each session ≈ 3 min: 1 min eyes-open relaxation, 1 min concentration (problems/object relations), 1 min eyes-closed relaxation.
- **License:** CC BY 4.0.

**Use this first** for “State of Mind” and a simple classifier (relaxation vs concentration).

---

### 2. **EEG Recordings Dataset for Mental Stress Detection** (stress vs relaxation)

- **Where:** [Mendeley Data – wnshbvdxs2](https://data.mendeley.com/datasets/wnshbvdxs2)
- **Why:** **Stress induction** (math, Trier, Stroop, horror video) plus **relaxing music**; good for stress vs relaxation.
- **Content:** 20–24 subjects, EMOTIV 5-channel sensor; multiple conditions per subject.
- **License:** CC BY 4.0.

**Use this** when you want to add “stress” as a state and keep the pipeline EMOTIV-friendly.

---

### 3. **PhysioNet EEG Motor Movement/Imagery** (fallback for pipeline testing)

- **Where:** [PhysioNet – eegmmidb](https://physionet.org/content/eegmmidb/1.0.0/)
- **Why:** Very well documented; loadable with **MNE-Python**; good to validate your pipeline (EDF, channel layout, band power) even though the labels are motor tasks, not “focus/relaxation.”
- **Format:** EDF, 64 channels, 160 Hz.

Use this if you want to get the **technical pipeline** (load → band power → API) working before dealing with Mendeley download/formats.

---

## Process: from dataset to dashboard

### Step 1 – Get the data

1. **Relaxation/Concentration (Mendeley 8c26dn6c7w)**  
   - Go to the Mendeley link, sign in if needed, click **Download dataset**.  
   - You’ll get a ZIP. Unzip and note the structure (e.g. one folder per subject, files per session).  
   - Check the **README or paper** on the dataset page for exact file format (CSV, EDF, or other) and column/channel names.

2. **Stress (Mendeley wnshbvdxs2)**  
   - Same idea: download from Mendeley, unzip, read the included description for format and labels.

3. **PhysioNet (optional)**  
   - Can be downloaded programmatically with MNE (see Step 2). No manual ZIP needed for a quick test.

---

### Step 2 – Load data in Python

- **If the dataset is EDF (or EDF+):** use **MNE**:
  ```bash
  pip install mne
  ```
  ```python
  import mne
  raw = mne.io.read_raw_edf("path/to/file.edf", preload=True)
  # raw.info["sfreq"], raw.ch_names, raw.get_data()
  ```

- **If the dataset is CSV/text:** use **pandas** (and maybe **NumPy**):
  - Read with `pd.read_csv()` (or similar), identify columns: time index, channel names (e.g. AF3, F7, … for EMOTIV).
  - Convert to a 2D array `(n_channels, n_samples)` and a sampling rate variable so the rest of the pipeline stays the same.

- **PhysioNet Motor/Imagery (EDF):**
  ```python
  from mne.datasets import eegbci
  from mne.io import read_raw_edf
  # Download and get file paths (subject 1, runs 6,10,14)
  files = eegbci.load_data(subject=1, runs=[6, 10, 14])
  raws = [read_raw_edf(f, preload=True) for f in files]
  ```

Create a small **loader module** (e.g. `data/load_*.py` or `pipeline/loaders.py`) that returns a common structure: **data array (channels × time), sampling rate, and labels/segment info** (e.g. “relaxation”, “concentration”, “stress”) so the rest of the pipeline is dataset-agnostic.

---

### Step 3 – Build a small data pipeline

Suggested layout:

```
data/                 # or eeg_data/ (add to .gitignore if large)
  ...                 # raw downloads / unzipped files
pipeline/
  loaders.py          # load_mendeley_relaxation(), load_physionet_eeg(), etc.
  features.py         # band power, PSD (delta, theta, alpha, beta, gamma)
  classifier.py      # simple model (e.g. LogisticRegression on band-power features)
  cache.py            # optional: save precomputed features/labels for fast API
```

- **loaders:** For each dataset, return:
  - `data`: `(n_channels, n_samples)` or `(n_segments, n_channels, n_samples)` if you use epochs.
  - `sfreq`: sampling rate in Hz.
  - `labels`: per segment or per file (e.g. `"relaxation"`, `"concentration"`, `"stress"`).
  - Optional: `channel_names`, `duration_sec`.

- **features:** From each segment (or a sliding window), compute **band power** (or PSD then integrate in bands):
  - **Delta** 1–4 Hz, **Theta** 4–8 Hz, **Alpha** 8–13 Hz, **Beta** 13–30 Hz, **Gamma** 30–45 Hz (or 30–80 Hz).
  - Use **Welch PSD** (`scipy.signal.welch`) or **MNE** (`mne.time_frequency.psd_array_welch` or `psd_array_multitaper`), then integrate power in each band (e.g. `scipy.integrate.simpson`).
  - Output: one feature vector per segment/window (e.g. 5 bands × N channels, or averaged over channels).

- **classifier:**  
  - Train a **Logistic Regression** (or small MLP) on band-power features with labels from the dataset.  
  - Save the model (e.g. `joblib` or `sklearn` pickle) and a small config (band edges, channel list) so the API can load it.

- **cache (optional):** Precompute band power and labels for all files; the API then loads from cache instead of raw EDF/CSV for speed.

---

### Step 4 – Expose the same API shape for the dashboard

Keep the **existing dashboard API contract** so the frontend does not need to change:

- **Time-series for the graph:**  
  For “current” or “selected” segment, provide:
  - `time`: time axis in seconds.
  - `alpha`, `beta`, `gamma`, `delta`: one array each (e.g. from a single channel or channel-averaged band-filtered signal, or from PSD-derived band power repeated over time for that segment).

So the API response stays:

```json
{
  "time": [...],
  "alpha": [...],
  "beta": [...],
  "gamma": [...],
  "delta": [...]
}
```

- **State of Mind / explanation:**  
  Add a second endpoint (e.g. `GET /api/state` or part of the same response) that returns:
  - `state`: e.g. `"relaxation"`, `"concentration"`, `"stress"`.
  - `confidence`: optional.
  - `explanation`: short text (e.g. “High alpha, low beta – consistent with relaxed state.”).

Implementation:

- In `main.py`, replace (or branch) the current `generate_sample_eeg()` path with:
  - Loading a segment from your **pipeline cache** or **live stream**.
  - Computing band power for that segment; optionally band-pass filtering and using the filtered signals as `alpha`/`beta`/`gamma`/`delta` time series for the graph.
- Run the **trained classifier** on the same segment’s band-power features and map the predicted class + confidence to `state` and `explanation`.

---

### Step 5 – Refine frontend “State of Mind” and explanation

- **State of Mind:**  
  Consume the new `state` (and optional `confidence`) from the API and show it in the existing “State of Mind” section (e.g. “Relaxation” / “Concentration” / “Stress” plus a short confidence line).

- **Explain the state:**  
  Use the `explanation` string from the API (e.g. “High alpha, low beta – consistent with relaxed state”) so the section is **model-driven** instead of static text.

No change to the graph contract; only the **source** of the JSON (real data + pipeline instead of synthetic) and the extra state/explanation fields.

---

## Quick reference: band power with MNE

```python
import mne
import numpy as np
from scipy.integrate import simpson

def band_power_from_raw(raw, bands=None):
    """raw: mne.io.Raw, bands: dict e.g. {'alpha': (8, 13), 'beta': (13, 30)}"""
    if bands is None:
        bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
    data, sfreq = raw.get_data(), raw.info["sfreq"]
    psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq, fmin=1, fmax=45)
    # psds shape: (n_channels, n_freqs)
    result = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        result[name] = simpson(psds[:, mask], freqs[mask], axis=1).mean()  # mean over channels
    return result
```

You can then use `result` as features for the classifier and, if needed, build simple time series for the graph (e.g. repeated constant or smoothed band power over the segment length).

---

## Summary

| Step | Action |
|------|--------|
| 1 | Choose **Relaxation/Concentration** (8c26dn6c7w) and/or **Stress** (wnshbvdxs2); optionally use PhysioNet for pipeline testing. |
| 2 | Download from Mendeley (and/or use MNE for PhysioNet); inspect format (EDF vs CSV). |
| 3 | Implement **loaders** → common structure (data, sfreq, labels). |
| 4 | Implement **band power** in `features.py`; then **train a small classifier** in `classifier.py`. |
| 5 | Wire pipeline into **`/api/eeg/sample`** (and new state/explanation endpoint) keeping the same JSON shape for the graph. |
| 6 | Update dashboard **State of Mind** and **Explain the state** to use API-driven state and explanation. |

Once this is in place, you can swap in your own EMOTIV recordings by adding another loader that outputs the same (data, sfreq, labels) and reusing the same feature and classifier pipeline.

---

## Pipeline code in this repo

- **`pipeline/loaders.py`** – `load_synthetic_segment()`, `load_emotiv_edf()`, `emotiv_edf_to_band_time_series()`, and `list_emotiv_edf_files()` for your **Emotiv Sample Data** EDFs. Data lives under `Emotiv Sample Data/S001/S001E01.edf` (subject S001, session E01), etc.
- **`pipeline/features.py`** – `band_power_psd()` and `band_power_time_series()` using scipy (Welch PSD + Simpson integration).
- **`pipeline/state_classifier.py`** – rule-based mental-state classifier: `classify(features)` returns predicted_state, confidence, explanation, scores, features. Replaceable later by a trained ML model.

---

## Mental-state inference (rule-based)

The dashboard infers a **State of Mind** from EEG band data using a simple rule-based classifier.

**Flow:** Load EDF → band time series → **feature_extractor** (mean power, relative power, beta/alpha, theta/beta, alpha/(beta+gamma)) → **state_classifier** → JSON with `predicted_state`, `confidence`, `explanation`, `scores`, `features`.

**Backend modules:**
- **`pipeline/eeg_loader.py`** – `list_emotiv_edf_files`, `get_edf_path`, `load_band_time_series`, `load_band_time_series_for_subject_session`.
- **`pipeline/feature_extractor.py`** – `extract_features(band_series)` returns mean/relative power and ratios.
- **`pipeline/state_classifier.py`** – `classify(features)` returns Relaxed / Focused / Drowsy / Stressed / Neutral/Unclear with heuristics (e.g. relaxed = high alpha, low beta/gamma; focused = elevated beta, low theta; drowsy = high theta/delta, low beta; stressed = high beta/gamma, low alpha).

**API:** `GET /api/state?subject=1&session=1&max_duration_sec=60` returns the inference JSON. The frontend calls this when the user clicks Load (same subject/session as the graph) and renders the result in the “State of Mind” and “Explain the state” sections, plus an optional disclaimer.

---

## Using your Emotiv Sample Data (EDF) in the dashboard

1. **Install MNE** (if not already): `pip install mne`
2. **Place EDFs** under `Emotiv Sample Data/S001/S001E01.edf`, etc. (you already have this.)
3. **Use real data in the API:**
   - **Synthetic (default):** `GET /api/eeg/sample` or `?seconds=10`
   - **Real EDF:** `GET /api/eeg/sample?use_real=1` (default: subject 1, session 1, first 60 s)
   - **Options:** `?use_real=1&subject=5&session=2&max_duration_sec=30`
4. **List files:** `GET /api/eeg/files` returns all detected (subject, session, path).
5. **Frontend:** Point the dashboard at real data by changing the fetch URL in `static/dashboard.js` to include `use_real=1` (and optional subject/session), or add a small “Use real data” toggle that switches the query params.
