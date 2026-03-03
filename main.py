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
    """
    EEG band time series from Emotiv Sample Data (EDF).
    subject (1–30), session (1–4), max_duration_sec, sample_rate (downsampling).
    """
    try:
        from pipeline.loaders import (
            list_emotiv_edf_files,
            emotiv_edf_to_band_time_series,
            DEFAULT_DATA_DIR,
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Pipeline/EDF not available: {e}") from e
    files = list_emotiv_edf_files(DEFAULT_DATA_DIR)
    if not files:
        raise HTTPException(
            status_code=404,
            detail="No EDF files found. Place data under 'Emotiv Sample Data/S001/S001E01.edf' etc.",
        )
    subj_id = f"S{subject:03d}"
    sess_id = f"E{session:02d}"
    path = None
    for s, e, p in files:
        if s == subj_id and e == sess_id:
            path = p
            break
    if path is None:
        path = files[0][2]
    out = emotiv_edf_to_band_time_series(
        path,
        max_duration_sec=max_duration_sec if max_duration_sec > 0 else None,
        target_sfreq=min(sample_rate, 250),
    )
    return out


@app.get("/api/eeg/files")
async def list_eeg_files():
    """List available real EDF files (subject, session, path) for the dashboard."""
    try:
        from pipeline.loaders import list_emotiv_edf_files, DEFAULT_DATA_DIR
    except ImportError:
        return {"files": [], "error": "Pipeline not available"}
    files = list_emotiv_edf_files(DEFAULT_DATA_DIR)
    return {
        "files": [
            {"subject": s, "session": e, "path": str(p)}
            for s, e, p in files
        ],
    }


@app.get("/", response_class=HTMLResponse)
async def show_dashboard(request: Request):
    return templates.TemplateResponse(
        "brain_state_dashboard.html",
        {"request": request},
    )
