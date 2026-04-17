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


def _load_pipeline_attr(module_name: str, attr_name: str):
    try:
        module = __import__(module_name, fromlist=[attr_name])
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline not available: {exc}") from exc
    return getattr(module, attr_name)


def _serialize_training_result(result: dict) -> dict:
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


@app.get("/api/model/status")
async def model_status():
    return _load_pipeline_attr("pipeline.state_classifier", "get_model_status")()


@app.get("/api/model/imagery/status")
async def imagery_model_status():
    return _load_pipeline_attr("pipeline.state_classifier", "get_second_model_status")()


@app.post("/api/model/train")
async def train_first_model(model: str = "svm"):
    run_training = _load_pipeline_attr("pipeline.state_classifier", "train_first_model")
    try:
        result = run_training(model_name=model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc
    return _serialize_training_result(result)


@app.post("/api/model/imagery/train")
async def train_second_model(model: str = "csp_lda"):
    run_training = _load_pipeline_attr("pipeline.state_classifier", "train_second_model")
    try:
        result = run_training(model_name=model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc
    return _serialize_training_result(result)


@app.get("/api/sessions")
async def list_sessions(subject: int = 1):
    list_shin2017_sessions = _load_pipeline_attr("pipeline.eeg_loader", "list_shin2017_sessions")
    try:
        sessions = list_shin2017_sessions(subject=subject, kind="state")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to list sessions: {exc}") from exc

    return {"subject": subject, "dataset": "Shin2017B", "sessions": sessions}


@app.get("/api/session/playback")
async def session_playback(subject: int = 1, session: int = 1):
    predict_session_timeline = _load_pipeline_attr("pipeline.state_classifier", "predict_session_timeline")
    try:
        return predict_session_timeline(subject=subject, session=session)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to score session: {exc}") from exc


@app.get("/api/session/imagery/playback")
async def imagery_session_playback(subject: int = 1, session: int = 1):
    predict_imagery_session_timeline = _load_pipeline_attr(
        "pipeline.state_classifier",
        "predict_imagery_session_timeline",
    )
    try:
        return predict_imagery_session_timeline(subject=subject, session=session)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to score imagery session: {exc}") from exc


@app.get("/", response_class=HTMLResponse)
async def show_dashboard(request: Request):
    return templates.TemplateResponse("brain_state_dashboard.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
