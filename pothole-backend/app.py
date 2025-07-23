"""
Flask App for Pothole Detection CV Pipeline
Clean version with utilities separated
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime
import uuid

# Import our simplified models
from models import load_midas_model, get_depth_map, segment_pothole, calculate_volume, determine_severity, save_3d_visualization

# Import utility functions
from utils import (
    load_existing_data, 
    save_data_to_json, 
    save_processed_images,
    save_processed_images2,
    save_uploaded_image,
    load_and_preprocess_image,
    calculate_statistics,
    create_directories,
    convert_numpy_types,
    get_proccessed_depth_map
)

app = Flask(__name__)
CORS(app)

# Create necessary directories
create_directories()

@app.route('/')
def root():
    """API information and available endpoints"""
    return {
        "message": "Pothole Detection CV Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "GET /": "API information and available endpoints",
            "GET /api/health": "Health check and model status", 
            "POST /api/upload": "Upload image for pothole detection",
            "GET /api/potholes": "Get all detected potholes",
            "GET /api/pothole/<id>": "Get specific pothole details",
            "GET /api/stats": "Get detection statistics"
        }
    }

@app.route('/api/health')
def health_check():
    """Health check and model status"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + 'Z',
        "models": {
            "depth_estimation": "ready",
            "segmentation": "ready",
            "volume_calculator": "ready"
        }
    }

@app.route('/api/potholes')
def get_potholes():
    """Get all detected potholes"""
    data = load_existing_data()
    return jsonify(data['potholes'])

@app.route('/api/upload', methods=['POST'])
def upload():
    """Upload image for pothole detection"""
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate unique ID
        pothole_id = f"PH{str(uuid.uuid4())[:6].upper()}"
        
        # Save uploaded image
        image_path, image_filename = save_uploaded_image(file, pothole_id)
        
        # Load and preprocess image
        image_rgb = load_and_preprocess_image(image_path)

        if image_rgb is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        print(f"Processing image: {image_filename}")
        
        # Step 1: Depth Estimation
        print("  → Estimating depth...")
        midas, transform, device = load_midas_model()

        depth_map, depth_vis = get_depth_map(image_rgb, midas, transform, device )
        
        # Step 2: Segmentation  
        print("  → Segmenting pothole...")
        binary_mask = segment_pothole("./model/best.pt",image_rgb)
        confidence = 0.78
        
        # Check if pothole detected
        if np.sum(binary_mask) == 0:
            return jsonify({'error': 'No pothole detected in image'}), 400
        
        # Step 3: Volume Calculation
        print("  → Calculating volume...")

        processed_depth_map = get_proccessed_depth_map(depth_vis)
        volume_results = calculate_volume(processed_depth_map, binary_mask)
        
        if volume_results is None:
            return jsonify({'error': 'Volume calculation failed'}), 400
        
        # Step 4: Determine Severity
        severity = determine_severity(volume_results['volume_mm3'], volume_results['max_depth_mm'])
        
        # Step 5: Save 3D Model
        print("  → Saving 3D model...")
        model_path = save_3d_visualization(depth_vis, pothole_id)
        
        # Step 6: Save Processed Images
        # pothole_id='4444'
        processed_images = save_processed_images2(pothole_id, depth_vis, binary_mask, image_rgb)
        
        # Get GPS coordinates
        latitude = float(request.form.get('latitude', 40.7128 + np.random.uniform(-0.01, 0.01)))
        longitude = float(request.form.get('longitude', -74.006 + np.random.uniform(-0.01, 0.01)))
        
        # Create pothole data - Convert numpy types to Python types for JSON serialization
        pothole_data = {
            'id': pothole_id,
            'latitude': latitude,
            'longitude': longitude,
            'depth': volume_results['max_depth_cm'],
            'volume': volume_results['volume_cm3'],
            'dimensions': volume_results,
            'severity': severity,
            'confidence': confidence,
            'description': f"Pothole detected - {severity} severity",
            'detectedAt': datetime.utcnow().isoformat() + 'Z',
            'imageUrl': f"/uploads/{image_filename}",
            'modelPath': model_path,
            'processedImages': processed_images,
            'volumeCm3': volume_results['volume_cm3']
        }
        
        # Ensure all numpy types are converted to JSON-serializable types
        pothole_data = convert_numpy_types(pothole_data)
        
        # Save to database
        save_data_to_json(pothole_data)
        
        print(f"  ✅ Completed: {severity} severity, {volume_results['volume_cm3']:.1f} cm³")
        
        return jsonify(pothole_data)
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/pothole/<pothole_id>')
def get_pothole_details(pothole_id):
    """Get specific pothole details"""
    data = load_existing_data()
    pothole = next((p for p in data["potholes"] if p["id"] == pothole_id), None)
    
    if not pothole:
        return jsonify({'error': 'Pothole not found'}), 404
    
    return jsonify(pothole)

@app.route('/api/stats')
def get_stats():
    """Get detection statistics"""
    data = load_existing_data()
    potholes = data["potholes"]
    return jsonify(calculate_statistics(potholes))

if __name__ == "__main__":
    app.run(debug=True, port=5000)