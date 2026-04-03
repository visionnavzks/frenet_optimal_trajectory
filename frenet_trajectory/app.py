import gradio as gr
import numpy as np
import ast
import matplotlib
matplotlib.use('Agg')
from frenet_optimal_trajectory import run_simulation

def generate_trajectory(obstacles_str, initial_speed, target_speed, view_mode):
    # Parse obstacles from string representation to numpy array
    try:
        obstacles_list = ast.literal_eval(obstacles_str)
        obstacles = np.array(obstacles_list)
    except Exception as e:
        return None, f"Error parsing obstacles: {e}"

    # Generate simulation and get the result path
    try:
        result_image_path, status = run_simulation(obstacles, initial_speed, target_speed, view_mode)
        import os
        return os.path.abspath(result_image_path), status
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Simulation Error: {e}"

# Default values from the original code (high speed)
default_obstacles = "[[3.0, 1.0], [5.0, -0.0], [6.0, 0.5], [8.0, -1.5]]"
default_initial_speed = 10.0 / 3.6  # m/s
default_target_speed = 30.0 / 3.6   # m/s

with gr.Blocks(title="Frenet Optimal Trajectory Generator") as interface:
    gr.Markdown("# Frenet Optimal Trajectory Generator")
    gr.Markdown("Simulates the optimal trajectory for a dynamic street scenario using a Frenet frame. Based on PythonRobotics.")

    with gr.Row():
        with gr.Column(scale=1):
            obstacles_input = gr.Textbox(label="Obstacles (e.g., [[x1, y1], [x2, y2]])", value=default_obstacles)
            initial_speed_input = gr.Slider(minimum=0.0, maximum=30.0, label="Initial Speed (m/s)", value=default_initial_speed)
            target_speed_input = gr.Slider(minimum=0.0, maximum=30.0, label="Target Speed (m/s)", value=default_target_speed)
            view_mode_input = gr.Radio(choices=["居中", "跟随车"], label="视图模式 (View Mode)", value="跟随车")
            submit_btn = gr.Button("Generate Trajectory")

        with gr.Column(scale=2):
            status_output = gr.Textbox(label="行为状态 (Behavior Status)", interactive=False)
            image_output = gr.Image(type="filepath", label="Frenet Trajectory Simulation")

    submit_btn.click(
        fn=generate_trajectory,
        inputs=[obstacles_input, initial_speed_input, target_speed_input, view_mode_input],
        outputs=[image_output, status_output]
    )

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
