from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import sys

# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frenet_optimal_trajectory import (
    run_simulation,
    ScenarioConfig,
    LateralMovement,
    LongitudinalMovement
)

app = FastAPI(title="Frenet Optimal Trajectory API")

class SimulationRequest(BaseModel):
    lateral_movement: str = "HIGH_SPEED"
    longitudinal_movement: str = "VELOCITY_KEEPING"

@app.post("/simulate")
def simulate(req: SimulationRequest):
    # Convert string to enums
    lat_enum = LateralMovement.HIGH_SPEED if req.lateral_movement == "HIGH_SPEED" else LateralMovement.LOW_SPEED
    lon_enum = LongitudinalMovement.VELOCITY_KEEPING if req.longitudinal_movement == "VELOCITY_KEEPING" else LongitudinalMovement.MERGING_AND_STOPPING

    config = ScenarioConfig(
        lateral_movement=lat_enum,
        longitudinal_movement=lon_enum
    )

    return run_simulation(config)

# Mount static files at root
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)