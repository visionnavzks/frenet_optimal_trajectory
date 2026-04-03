import sys
sys.path.append("/app/frenet_trajectory")
from frenet_optimal_trajectory import run_simulation, ScenarioConfig

config = ScenarioConfig()
result = run_simulation(config)
print("Keys:", result.keys())
print("Target course length:", len(result['target_course_x']))
print("Obstacles length:", len(result['obstacles']))
print("Trace length:", len(result['trace']))
print("First trace element keys:", result['trace'][0].keys() if len(result['trace']) > 0 else "No trace")
