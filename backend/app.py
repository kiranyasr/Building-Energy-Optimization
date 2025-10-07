# =======================================================================
# Main Flask Application for the Energy Analysis Pipeline
# =======================================================================
import os
import traceback
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# --- Custom Function Imports ---
from data_handler import load_and_preview_data
from data_cleaner import clean_and_summarize_data
from feature_engineering import analyze_fluctuations
from advanced_features import create_engineered_features
from feature_selection import select_and_reduce_features
from baseline_modeling import train_and_evaluate_models
from deep_learning_models import train_advanced_models
from optimization import run_optimization
from anomaly_detection import detect_anomalies
from wastage_analysis import analyze_wastage_and_usage
from reporting import compile_final_report
from deployment_logic import get_feature_list, make_prediction

# --- Flask App Initialization ---
app = Flask(
    __name__,
    template_folder='../frontend/templates',
    static_folder='../frontend/static'
)

# --- Configuration ---
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024  # 128 MB
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'data')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Helper for JSON Serialization ---
def to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    return obj

# =======================================================================
# ROUTES
# =======================================================================

@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_data.csv')
        file.save(filepath)
        preview = load_and_preview_data(filepath)
        return jsonify(to_serializable(preview))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- Cleaning ---
@app.route('/clean')
def cleaning_page():
    return render_template('clean.html')


@app.route('/perform_cleaning', methods=['POST'])
def perform_cleaning():
    try:
        result = clean_and_summarize_data()
        return jsonify(to_serializable(result))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- Fluctuations ---
@app.route('/fluctuations')
def fluctuations_page():
    return render_template('fluctuations.html')


@app.route('/get_fluctuation_data', methods=['POST'])
def get_fluctuation_data():
    try:
        data = analyze_fluctuations()
        return jsonify(to_serializable(data))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- Feature Engineering ---
@app.route('/features')
def features_page():
    return render_template('features.html')


@app.route('/get_engineered_features', methods=['POST'])
def get_engineered_features():
    try:
        features = create_engineered_features()
        return jsonify(to_serializable(features))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- Feature Selection ---
@app.route('/selection')
def selection_page():
    return render_template('selection.html')


@app.route('/get_selection_data', methods=['POST'])
def get_selection_data():
    try:
        results = select_and_reduce_features()
        return jsonify(to_serializable(results))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- Baseline Models ---
@app.route('/models')
def models_page():
    return render_template('models.html')


@app.route('/run_baseline_models', methods=['POST'])
def run_baseline_models():
    try:
        results = train_and_evaluate_models()
        return jsonify(to_serializable(results))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


# --- Deep Learning Models ---
@app.route('/deep_learning')
def deep_learning_page():
    return render_template('deep_learning.html')


@app.route('/run_advanced_models', methods=['POST'])
def run_advanced_models():
    try:
        data = train_advanced_models()
        return jsonify(to_serializable(data))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


# --- Optimization ---
@app.route('/optimization')
def optimization_page():
    return render_template('optimization.html')


@app.route('/run_optimization', methods=['POST'])
def run_optimization_route():
    try:
        data = run_optimization()
        return jsonify(to_serializable(data))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- Anomaly Detection ---
@app.route('/anomalies')
def anomalies_page():
    return render_template('anomalies.html')


@app.route('/run_anomaly_detection', methods=['POST'])
def run_anomaly_detection_route():
    try:
        data = detect_anomalies()
        return jsonify(to_serializable(data))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- Wastage Analysis ---
@app.route('/wastage')
def wastage_page():
    return render_template('wastage.html')


@app.route('/run_wastage_analysis', methods=['POST'])
def run_wastage_analysis_route():
    try:
        data = analyze_wastage_and_usage()
        return jsonify(to_serializable(data))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- Final Report / Comparison ---
@app.route('/comparison')
def comparison_page():
    return render_template('comparison.html')


@app.route('/get_comparison_data', methods=['POST'])
def get_comparison_data_route():
    try:
        data = compile_final_report()
        if "error" not in data:
            data["artifacts"] = [{"name": "Comparison Summary Excel", "path": "comparison_summary.xlsx"}]
        return jsonify(to_serializable(data))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download/<path:filename>')
def download_file_route(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404


# --- Deployment ---
@app.route('/deployment')
def deployment_page():
    features = get_feature_list()
    return render_template('deployment.html', features=features)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json.get('features', [])
        prediction = make_prediction(input_data)
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


# --- User Guide ---
@app.route('/user_guide')
def user_guide_page():
    return render_template('user_guide.html')


# =======================================================================
# Main Execution
# =======================================================================
if __name__ == '__main__':
    app.run(debug=True)
