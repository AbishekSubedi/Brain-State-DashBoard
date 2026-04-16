from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI(
    title="Brain State Dashboard",
    description="Train a Shin2017 brain-state model and animate session-level predictions.",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/api/model/status")
async def model_status():
    try:
        from pipeline.state_classifier import get_model_status
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline not available: {exc}") from exc
    return get_model_status()


@app.post("/api/model/train")
async def train_first_model(model: str = "svm"):
    try:
        from pipeline.state_classifier import train_first_model as run_training
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline not available: {exc}") from exc

    try:
        result = run_training(model_name=model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    return {
        "trained": True,
        "artifact_dir": result["artifact_dir"],
        "metadata": result["metadata"],
        "metrics": {
            "accuracy": result["metrics"]["accuracy"],
            "confusion_matrix": result["metrics"]["confusion_matrix"].tolist(),
            "classification_report": result["metrics"]["classification_report"],
            "cross_val_accuracy_mean": round(float(result["cv_scores"].mean()), 4),
            "cross_val_accuracy_std": round(float(result["cv_scores"].std()), 4),
        },
    }


@app.get("/api/sessions")
async def list_sessions(subject: int = 1):
    try:
        from pipeline.eeg_loader import list_shin2017_sessions
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline not available: {exc}") from exc

    try:
        sessions = list_shin2017_sessions(subject=subject, kind="state")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to list sessions: {exc}") from exc

    return {"subject": subject, "dataset": "Shin2017B", "sessions": sessions}


@app.get("/api/session/playback")
async def session_playback(subject: int = 1, session: int = 1):
    try:
        from pipeline.state_classifier import predict_session_timeline
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline not available: {exc}") from exc

    try:
        return predict_session_timeline(subject=subject, session=session)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to score session: {exc}") from exc


@app.get("/", response_class=HTMLResponse)
async def show_dashboard(request: Request):
    return templates.TemplateResponse("brain_state_dashboard.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
