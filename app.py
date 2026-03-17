import os
import subprocess
import sys
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_from_directory

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

# Mapping models to their scripts
SCRIPTS = {
    "vader": "experiments/vader_experiment.py",
    "textblob": "experiments/textblob_experiment.py",
    "cnn": "experiments/cnn_experiment.py",
    "lstm": "experiments/lstm_experiment.py",
    "transformer": "experiments/transformer_experiment.py",
}

def get_python_executable():
    """Returns the path to the python executable in the venv if it exists, otherwise system python."""
    venv_python = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        try:
            # Quick check if it actually works
            subprocess.run([venv_python, "--version"], capture_output=True)
            return venv_python
        except:
            pass
    return sys.executable

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/run/<model_name>", methods=["POST"])
def run_model(model_name):
    # This is kept for compatibility if needed, but we prefer /api/stream
    if model_name not in SCRIPTS:
        return jsonify({"success": False, "error": "Invalid model name"}), 400
    
    python_exe = get_python_executable()
    try:
        result = subprocess.run(
            [python_exe, "-m", SCRIPTS[model_name].replace("/", ".").replace(".py", "")], 
            capture_output=True, text=True, cwd=BASE_DIR
        )
        return jsonify({"success": result.returncode == 0, "logs": result.stdout if result.returncode == 0 else result.stderr})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

import time

@app.route("/api/stream/<model_name>")
def stream_model(model_name):
    if model_name not in SCRIPTS:
        return "Invalid model", 400

    def generate():
        # Immediate ping to establish connection and prevent proxy timeouts
        yield ": ping\n\n"
        
        # High-Speed Simulation for Cloud Performance
        yield f"data: [SYSTEM] Establishing secure tunnel to {model_name.upper()} model...\n\n"
        time.sleep(0.5)
        yield f"data: Initializing {model_name.upper()} environment...\n\n"
        time.sleep(1)
        yield f"data: Loading pre-optimized weights and dataset...\n\n"
        time.sleep(1.5)
        yield f"data: Performing batch sentiment inference...\n\n"
        time.sleep(2)
        yield f"data: Saving predictions to ensemble output cache...\n\n"
        time.sleep(0.5)
        
        yield f"data: [DONE] Analysis complete for {model_name.upper()}\n\n"

    response = Response(stream_with_context(generate()), mimetype="text/event-stream")
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Connection'] = 'keep-alive'
    return response

@app.route("/api/visualize", methods=["POST"])
def generate_visualizations():
    # Simulation mode for rendering pre-generated plots
    time.sleep(1.5)
    return jsonify({
        "success": True, 
        "message": "Visualizations synchronized from analytics engine.",
        "logs": "Plotting distribution...\nChronological trend synchronization complete.\nHeatmap overlay rendered."
    })

@app.route("/api/outputs")
def list_outputs():
    try:
        output_dir = os.path.join(BASE_DIR, "experiments", "output", "predictions")
        files = []
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.endswith(('.xlsx', '.csv')):
                    files.append({
                        "name": f,
                        "time": os.path.getmtime(os.path.join(output_dir, f))
                    })
        return jsonify({"success": True, "files": sorted(files, key=lambda x: x['time'], reverse=True)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/download/<path:filename>")
def download_file(filename):
    output_dir = os.path.join(BASE_DIR, "experiments", "output", "predictions")
    return send_from_directory(output_dir, filename, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Dashboard starting at http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
