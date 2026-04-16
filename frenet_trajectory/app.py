from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import os
import sys

# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frenet_optimal_trajectory import (
    iter_simulation_events,
    run_simulation,
    ScenarioConfig,
    LateralMovement,
    LongitudinalMovement
)

app = FastAPI(title="Frenet Optimal Trajectory API")

class SimulationRequest(BaseModel):
    lateral_movement: str = "HIGH_SPEED"
    longitudinal_movement: str = "VELOCITY_KEEPING"


def build_config(req: SimulationRequest) -> ScenarioConfig:
    lat_enum = LateralMovement.HIGH_SPEED if req.lateral_movement == "HIGH_SPEED" else LateralMovement.LOW_SPEED
    lon_enum = LongitudinalMovement.VELOCITY_KEEPING if req.longitudinal_movement == "VELOCITY_KEEPING" else LongitudinalMovement.MERGING_AND_STOPPING

    return ScenarioConfig(
        lateral_movement=lat_enum,
        longitudinal_movement=lon_enum
    )

@app.post("/simulate")
def simulate(req: SimulationRequest):
    config = build_config(req)
    return run_simulation(config)


@app.post("/simulate/stream")
def stream_simulate(req: SimulationRequest):
    config = build_config(req)

    def event_stream():
        for event in iter_simulation_events(config):
            yield json.dumps(event) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

# Mount static files at root
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)