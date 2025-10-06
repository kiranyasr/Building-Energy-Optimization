# ==============================================================================
# Main Flask Application for the Energy Analysis Pipeline
# ==============================================================================
# This script acts as the central hub for the web application. It defines all
# the API routes (endpoints) that connect the frontend UI with backend logic.
# When a button is clicked on the webpage, it triggers a request to one of these
# endpoints, which then calls the corresponding backend Python function and
# returns the processed results.
# ==============================================================================

# --- Core Libraries & Setup ---
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import traceback

# --- Custom Function Imports ---
# Each function below performs a major step in the ML & data pipeline.
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

# --- Configuration Settings ---
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024  # Max upload size = 128 MB
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'data')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==============================================================================
# API Routes for the Analysis Pipeline
# ==============================================================================

# --- 1. Data Upload & Preview ---
@app.route('/')
def index():
    """Renders the main landing page (upload.html)."""
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads, saves the file, and returns a data preview."""
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
        return jsonify(preview)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- 2. Data Cleaning & Preprocessing ---
@app.route('/clean')
def cleaning_page():
    """Renders the data cleaning page (clean.html)."""
    return render_template('clean.html')


@app.route('/perform_cleaning', methods=['POST'])
def perform_cleaning():
    """Triggers data cleaning and returns a summary."""
    try:
        result = clean_and_summarize_data()
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- 3. Fluctuation Analysis ---
@app.route('/fluctuations')
def fluctuations_page():
    """Renders the fluctuation analysis dashboard (fluctuations.html)."""
    return render_template('fluctuations.html')


@app.route('/get_fluctuation_data', methods=['POST'])
def get_fluctuation_data():
    """Triggers fluctuation analysis and returns chart data."""
    try:
        data = analyze_fluctuations()
        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- 4. Feature Engineering ---
@app.route('/features')
def features_page():
    """Renders the feature engineering page (features.html)."""
    return render_template('features.html')


@app.route('/get_engineered_features', methods=['POST'])
def get_engineered_features():
    """Triggers feature engineering and returns sample data."""
    try:
        features = create_engineered_features()
        return jsonify(features)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- 5. Feature Selection ---
@app.route('/selection')
def selection_page():
    """Renders the feature selection page (selection.html)."""
    return render_template('selection.html')


@app.route('/get_selection_data', methods=['POST'])
def get_selection_data():
    """Triggers feature selection and returns results."""
    try:
        results = select_and_reduce_features()
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- 6. Baseline Modeling ---
@app.route('/models')
def models_page():
    """Renders the baseline model comparison page (models.html)."""
    return render_template('models.html')


@app.route('/run_baseline_models', methods=['POST'])
def run_baseline_models():
    """Runs baseline ML models and returns performance results."""
    try:
        results = train_and_evaluate_models()
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


# --- 7. Deep Learning Modeling ---
@app.route('/deep_learning')
def deep_learning_page():
    """Renders the deep learning models page (deep_learning.html)."""
    return render_template('deep_learning.html')


@app.route('/run_advanced_models', methods=['POST'])
def run_advanced_models():
    """Runs deep learning models and returns their results."""
    try:
        data = train_advanced_models()
        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


# --- 8. Optimization ---
@app.route('/optimization')
def optimization_page():
    """Renders the optimization results page (optimization.html)."""
    return render_template('optimization.html')


@app.route('/run_optimization', methods=['POST'])
def run_optimization_route():
    """Runs optimization and returns comparative results."""
    try:
        data = run_optimization()
        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- 9. Anomaly Detection ---
@app.route('/anomalies')
def anomalies_page():
    """Renders the anomaly detection dashboard (anomalies.html)."""
    return render_template('anomalies.html')


@app.route('/run_anomaly_detection', methods=['POST'])
def run_anomaly_detection_route():
    """Triggers anomaly detection and returns the findings."""
    try:
        data = detect_anomalies()
        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- 10. Wastage Analysis ---
@app.route('/wastage')
def wastage_page():
    """Renders the wastage and usage analysis page (wastage.html)."""
    return render_template('wastage.html')


@app.route('/run_wastage_analysis', methods=['POST'])
def run_wastage_analysis_route():
    """Analyzes wastage and usage efficiency."""
    try:
        data = analyze_wastage_and_usage()
        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- 11. Final Reporting ---
@app.route('/comparison')
def comparison_page():
    """Renders the final model comparison page (comparison.html)."""
    return render_template('comparison.html')


@app.route('/get_comparison_data', methods=['POST'])
def get_comparison_data_route():
    """Compiles and returns the final report."""
    try:
        data = compile_final_report()
        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- 12. Deployment & Live Predictions ---
@app.route('/deployment')
def deployment_page():
    """Renders the live deployment page (deployment.html)."""
    features = get_feature_list()
    return render_template('deployment.html', features=features)


@app.route('/predict', methods=['POST'])
def predict():
    """Receives input features from frontend and returns a prediction."""
    try:
        input_data = request.json.get('features', [])
        prediction = make_prediction(input_data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


# --- 13. Static Pages & File Downloads ---
@app.route('/user_guide')
def user_guide_page():
    """Renders the static user guide page."""
    return render_template('user_guide.html')


@app.route('/download/<path:filename>')
def download_file_route(filename):
    """Securely serves files for download from the data directory."""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True)
