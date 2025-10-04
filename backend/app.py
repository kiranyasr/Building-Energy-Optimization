# app.py
import os
from flask import Flask, render_template, request, jsonify

# Import all custom functions
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


app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'data/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# All routes from Day 1 to 9
@app.route('/')
def index(): return render_template('upload.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_data.csv')
    file.save(filepath)
    try: return jsonify(load_and_preview_data(filepath))
    except Exception as e: return jsonify({'error': str(e)}), 500
@app.route('/clean')
def cleaning_page(): return render_template('clean.html')
@app.route('/perform_cleaning', methods=['POST'])
def perform_cleaning():
    try: return jsonify(clean_and_summarize_data())
    except Exception as e: return jsonify({'error': str(e)}), 500
@app.route('/fluctuations')
def fluctuations_page(): return render_template('fluctuations.html')
@app.route('/get_fluctuation_data', methods=['POST'])
def get_fluctuation_data():
    try: return jsonify(analyze_fluctuations())
    except Exception as e: return jsonify({'error': str(e)}), 500
@app.route('/features')
def features_page(): return render_template('features.html')
@app.route('/get_engineered_features', methods=['POST'])
def get_engineered_features():
    try: return jsonify(create_engineered_features())
    except Exception as e: return jsonify({'error': str(e)}), 500
@app.route('/selection')
def selection_page(): return render_template('selection.html')
@app.route('/get_selection_data', methods=['POST'])
def get_selection_data():
    try: return jsonify(select_and_reduce_features())
    except Exception as e: return jsonify({'error': str(e)}), 500
@app.route('/models')
def models_page(): return render_template('models.html')
@app.route('/run_baseline_models', methods=['POST'])
def run_baseline_models():
    try:
        results = train_and_evaluate_models()
        if "error" in results: return jsonify(results), 404
        return jsonify(results)
    except Exception as e: return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500
@app.route('/deep_learning')
def deep_learning_page(): return render_template('deep_learning.html')
@app.route('/run_advanced_models', methods=['POST'])
def run_advanced_models():
    try:
        data = train_advanced_models()
        if "error" in data: return jsonify(data), 404
        return jsonify(data)
    except Exception as e: return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500
@app.route('/optimization')
def optimization_page(): return render_template('optimization.html')
@app.route('/run_optimization', methods=['POST'])
def run_optimization_route():
    try:
        data = run_optimization()
        if "error" in data: return jsonify(data), 404
        return jsonify(data)
    except Exception as e: return jsonify({'error': str(e)}), 500
@app.route('/anomalies')
def anomalies_page(): return render_template('anomalies.html')
@app.route('/run_anomaly_detection', methods=['POST'])
def run_anomaly_detection_route():
    try:
        data = detect_anomalies()
        if "error" in data: return jsonify(data), 404
        return jsonify(data)
    except Exception as e: return jsonify({'error': str(e)}), 500

# Day 10 Route
@app.route('/wastage')
def wastage_page():
    return render_template('wastage.html')

@app.route('/run_wastage_analysis', methods=['POST'])
def run_wastage_analysis_route():
    try:
        data = analyze_wastage_and_usage()
        if "error" in data: return jsonify(data), 404
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Day 11 Route
@app.route('/comparison')
def comparison_page():
    """Renders the final model comparison and reporting page."""
    return render_template('comparison.html')


@app.route('/get_comparison_data', methods=['POST'])
def get_comparison_data_route():
    """Triggers the final reporting compilation."""
    try:
        data = compile_final_report()
        if "error" in data:
            return jsonify(data), 404
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)