# Brain-State-DashBoard

Brain-State-DashBoard is a web-based platform that visualizes EEG brain-wave signals and estimates cognitive states such as focus, relaxation, and stress.

This project supports my undergraduate research on brain waves and frequencies and is also being developed as part of a tech startup course. The goal is to combine lightweight machine learning with clear, interpretable visualizations to show how mental states change over time.

## Current Status
The first version of the web dashboard is running with a main page layout and a sample EEG visualization.  
Work currently focuses on wiring in real EEG data and adding basic cognitive state estimation.

## What’s Implemented Now
- Main dashboard page with:
  - Header (title + navigation)
  - “About the project” info section
  - EEG graph section with band toggles (Alpha, Beta, Gamma, Delta)
  - “State of Mind” and explanation area
- Sample EEG-like data served from a FastAPI endpoint at `/api/eeg/sample`
- Frontend graph rendered with Chart.js and a small JS module in `static/dashboard.js`

## Running the App
```bash
cd Brain-State-DashBoard
source venv/bin/activate        # or . venv/bin/activate
uvicorn main:app --reload
```
Then open `http://127.0.0.1:8000/` in your browser.

## Swapping to Real EEG Data
The endpoint `/api/eeg/sample` currently returns synthetic EEG-like band data:

```json
{
  "time": [...],
  "alpha": [...],
  "beta": [...],
  "gamma": [...],
  "delta": [...]
}
```

When you have real EMOTIV/EEG data, update the implementation of `generate_sample_eeg` / `/api/eeg/sample` in `main.py` to return the same JSON shape using your live or recorded signals. The frontend graph and layout will work without changes.

## Planned Features
- EEG signal visualization (time-domain and frequency bands)
- Cognitive state classification using simple machine learning models
- Timeline view showing changes in mental state over a session
- Web dashboard built with FastAPI and Jinja templates

## Tech Stack
- Python
- FastAPI
- Jinja2
- Lightweight machine learning (e.g., Logistic Regression / SVM)
- Consumer EEG data (EMOTIV headset)

## Notes
This project is for research, visualization, and educational purposes only and does not provide medical diagnosis or treatment.
