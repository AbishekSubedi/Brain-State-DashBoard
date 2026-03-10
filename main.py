import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Brain State Dashboard",
    description="A dashboard for visualizing brain state data.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/api/eeg/sample")
async def get_eeg_data(
    subject: int = 1,
    session: int = 1,
    max_duration_sec: float = 60.0,
    sample_rate: int = 128,
):
    """EEG band time series from Emotiv Sample Data (EDF)."""
    try:
        from pipeline.eeg_loader import (
            list_emotiv_edf_files,
            load_band_time_series_for_subject_session,
            DEFAULT_DATA_DIR,
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Pipeline not available: {e}") from e
    data = load_band_time_series_for_subject_session(
        subject, session,
        max_duration_sec=max_duration_sec if max_duration_sec > 0 else None,
        target_sfreq=min(sample_rate, 250),
        data_dir=DEFAULT_DATA_DIR,
    )
    if data is None:
        raise HTTPException(status_code=404, detail="No EDF files found.")
    return data


@app.get("/api/state")
async def get_state(subject: int = 1, session: int = 1, max_duration_sec: float = 60.0):
    """
    Mental state inference from EEG band features (rule-based).
    Returns predicted_state, confidence, explanation, scores, features.
    """
    try:
        from pipeline.eeg_loader import load_band_time_series_for_subject_session, DEFAULT_DATA_DIR
        from pipeline.feature_extractor import extract_features
        from pipeline.state_classifier import classify
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Pipeline not available: {e}") from e
    band_data = load_band_time_series_for_subject_session(
        subject, session,
        max_duration_sec=max_duration_sec if max_duration_sec > 0 else None,
        target_sfreq=128,
        data_dir=DEFAULT_DATA_DIR,
    )
    if band_data is None:
        raise HTTPException(status_code=404, detail="No EDF data for this subject/session.")
    features = extract_features(band_data)
    result = classify(features)
    return result


@app.get("/api/eeg/files")
async def list_eeg_files():
    """List available EDF files (subject, session, path)."""
    try:
        from pipeline.eeg_loader import list_emotiv_edf_files, DEFAULT_DATA_DIR
    except ImportError:
        return {"files": [], "error": "Pipeline not available"}
    files = list_emotiv_edf_files(DEFAULT_DATA_DIR)
    return {"files": [{"subject": s, "session": e, "path": str(p)} for s, e, p in files]}


@app.get("/", response_class=HTMLResponse)
async def show_dashboard(request: Request):
    return templates.TemplateResponse("brain_state_dashboard.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
