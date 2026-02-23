import uvicorn
import numpy as np
from fastapi import FastAPI, Request
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


def generate_sample_eeg(seconds: float = 10.0, sample_rate: int = 128) -> dict:
    """Generate synthetic EEG-like data for Alpha, Beta, Gamma, Delta bands.
    Replace this with real EMOTIV/EEG data when available."""
    n = int(seconds * sample_rate)
    t = np.linspace(0, seconds, n)
    # Band frequencies (Hz): Delta 0.5-4, Theta 4-8, Alpha 8-12, Beta 12-30, Gamma 30-45
    rng = np.random.default_rng(42)
    alpha = np.sin(2 * np.pi * 10 * t) * 0.5 + rng.normal(0, 0.05, n)
    beta = np.sin(2 * np.pi * 20 * t) * 0.4 + rng.normal(0, 0.05, n)
    gamma = np.sin(2 * np.pi * 38 * t) * 0.3 + rng.normal(0, 0.04, n)
    delta = np.sin(2 * np.pi * 2 * t) * 0.6 + rng.normal(0, 0.06, n)
    return {
        "time": t.tolist(),
        "alpha": alpha.tolist(),
        "beta": beta.tolist(),
        "gamma": gamma.tolist(),
        "delta": delta.tolist(),
    }


@app.get("/api/eeg/sample")
async def get_sample_eeg(seconds: float = 10.0, sample_rate: int = 128):
    """Sample EEG data for the graph. Swap this for real data source when ready."""
    return generate_sample_eeg(seconds=seconds, sample_rate=sample_rate)


@app.get("/", response_class=HTMLResponse)
async def show_dashboard(request: Request):
    return templates.TemplateResponse(
        "brain_state_dashboard.html",
        {"request": request},
    )
