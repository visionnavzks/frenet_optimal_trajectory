import gradio as gr
import numpy as np
import ast
import matplotlib
matplotlib.use('Agg')
from frenet_optimal_trajectory import run_simulation

def generate_trajectory(obstacles_str, initial_speed, target_speed):
    # Parse obstacles from string representation to numpy array
    try:
        obstacles_list = ast.literal_eval(obstacles_str)
        obstacles = np.array(obstacles_list)
    except Exception as e:
        return f"Error parsing obstacles: {e}"

    # Generate simulation and get the result path
    try:
        result_image_path = run_simulation(obstacles, initial_speed, target_speed)
        import os
        return os.path.abspath(result_image_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None # or handle error display if possible

# Default values from the original code (high speed)
default_obstacles = "[[3.0, 1.0], [5.0, -0.0], [6.0, 0.5], [8.0, -1.5]]"
default_initial_speed = 10.0 / 3.6  # m/s
default_target_speed = 30.0 / 3.6   # m/s

interface = gr.Interface(
    fn=generate_trajectory,
    inputs=[
        gr.Textbox(label="Obstacles (e.g., [[x1, y1], [x2, y2]])", value=default_obstacles),
        gr.Slider(minimum=0.0, maximum=30.0, label="Initial Speed (m/s)", value=default_initial_speed),
        gr.Slider(minimum=0.0, maximum=30.0, label="Target Speed (m/s)", value=default_target_speed)
    ],
    outputs=gr.Image(type="filepath", label="Frenet Trajectory Simulation"),
    title="Frenet Optimal Trajectory Generator",
    description="Simulates the optimal trajectory for a dynamic street scenario using a Frenet frame. Based on PythonRobotics.",
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
