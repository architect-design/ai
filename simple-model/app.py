# app.py
from flask import Flask, render_template, request, jsonify
from services import ModelTrainer, model_registry
import threading

app = Flask(__name__)

# --- LOAD EXISTING MODELS ON STARTUP ---
print("Checking for saved models...")
ModelTrainer.load_all_models()
print("Startup complete.")

# In-memory status tracking
training_status = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"})

    file = request.files['file']
    format_name = request.form.get('format_name')

    if file.filename == '' or not format_name:
        return jsonify({"status": "error", "message": "Missing file or format name"})

    try:
        # Read content immediately
        content = file.read().decode('utf-8')
        # Repeat content to ensure enough data for training
        training_content = content * 3

        # Define the background task
        def train_task():
            try:
                training_status[format_name] = "training"
                ModelTrainer.train_format(format_name, training_content)
                training_status[format_name] = "completed"
            except Exception as e:
                print(f"Error training {format_name}: {e}")
                training_status[format_name] = f"error: {str(e)}"

        # Initialize status and start thread
        training_status[format_name] = "starting"
        thread = threading.Thread(target=train_task)
        thread.start()

        return jsonify({"status": "started", "message": f"Training started for '{format_name}'..."})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/status/<format_name>', methods=['GET'])
def check_status(format_name):
    """Endpoint for the UI to check if training is finished."""
    status = training_status.get(format_name, "unknown")
    return jsonify({"format": format_name, "status": status})


# app.py (Only the generate route needs changing)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    format_name = data.get('format_name')
    start_str = data.get('start_str', '')

    # NEW: Accept amount and mode
    amount = int(data.get('amount', 300))
    mode = data.get('mode', 'chars')  # 'chars' or 'lines'

    if format_name not in model_registry:
        return jsonify({"status": "error", "message": "Model not trained for this format."})

    try:
        # Pass mode and amount to the service
        result = ModelTrainer.generate_text(format_name, start_str, amount, mode)
        return jsonify({"status": "success", "generated_text": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
@app.route('/list_models', methods=['GET'])
def list_models():
    # Return both the model names and their current training status
    models_data = []
    # We check model_registry for completed models
    for name in model_registry.keys():
        models_data.append({"name": name, "status": "ready"})

    # We check training_status for models currently processing
    for name, status in training_status.items():
        if status == "training" or status == "starting":
            models_data.append({"name": name, "status": "training"})

    return jsonify({"models": models_data})


if __name__ == '__main__':
    app.run(debug=True, port=5000)