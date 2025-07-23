"""
Utility functions for data and image handling
"""

import json
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def load_existing_data():
    """Load existing pothole data from JSON file"""
    try:
        with open("./data/potholes.json", 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        # Create initial data structure if file doesn't exist
        initial_data = {"potholes": [], "total_count": 0}
        with open("./data/potholes.json", 'w') as file:
            json.dump(initial_data, file, indent=2)
        return initial_data

def save_data_to_json(new_pothole_data):
    """Append new pothole data to JSON file"""
    data = load_existing_data()
    data["potholes"].append(new_pothole_data)
    data["total_count"] = len(data["potholes"])
    
    with open("./data/potholes.json", 'w') as file:
        json.dump(data, file, indent=2)

def save_processed_images(pothole_id, depth_map, binary_mask):
    """Save processed images for debugging"""
    processed_dir = f"./processed/{pothole_id}"
    os.makedirs(processed_dir, exist_ok=True)
    
    try:
        # Save depth map visualization
        depth_img = (depth_map).astype(np.uint8)
        cv2.imwrite(f"{processed_dir}/depth_map.png", depth_img)
        
        # Save segmentation mask
        mask_img = binary_mask * 255
        cv2.imwrite(f"{processed_dir}/mask.png", mask_img)
        
        # Save masked depth
        masked_depth = depth_map * binary_mask
        masked_depth_img = (masked_depth ).astype(np.uint8)
        cv2.imwrite(f"{processed_dir}/masked_depth.png", masked_depth_img)
        
        return {
            'depth_map': f"{processed_dir}/depth_map.png",
            'mask': f"{processed_dir}/mask.png",
            'masked_depth': f"{processed_dir}/masked_depth.png"
        }
    except Exception as e:
        print(f"Warning: Failed to save processed images: {e}")
        return {}

def save_processed_images2(pothole_id, depth_vis, mask, image):
    """Save visualization following code 1 pattern"""
    processed_dir = f"./processed/{pothole_id}"
    os.makedirs(processed_dir, exist_ok=True)
    
    try:
        # Create the 3-panel visualization (from code 2 approach)
        plt.figure(figsize=(15, 5))
        
        # Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Segmentation Mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Segmentation Mask")
        plt.axis('off')
        
        # Depth Map
        plt.subplot(1, 3, 3)
        plt.imshow(depth_vis, cmap='inferno')
        plt.title("Depth Map")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save it (following code 1 pattern - no show, just save)
        plt.savefig(f"{processed_dir}/visualization.png", dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        return {
            'visualization': f"{processed_dir}/visualization.png"
        }
        
    except Exception as e:
        print(f"Warning: Failed to save processed visualization: {e}")
        plt.close()  # Ensure figure is closed even on error
        return {}
    
def save_uploaded_image(file, pothole_id):
    """Save uploaded image file"""
    image_filename = f"{pothole_id}_original.jpg"
    image_path = f"./uploads/{image_filename}"
    file.save(image_path)
    return image_path, image_filename

def letterbox(image, size=(640, 640), color=(114, 114, 114)):
    h, w = image.shape[:2]
    scale = min(size[0] / h, size[1] / w)  # Uniform scale
    nh, nw = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (nw, nh))
    
    # Compute padding
    top = (size[0] - nh) // 2
    bottom = size[0] - nh - top
    left = (size[1] - nw) // 2
    right = size[1] - nw - left
    
    # Add padding
    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )
    return padded_image

def load_and_preprocess_image(image_path):
    """Load image and convert to RGB"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return letterbox(img_rgb)

def calculate_statistics(potholes):
    """Calculate statistics from pothole data"""
    if not potholes:
        return {
            'total': 0,
            'severity_distribution': {},
            'avg_volume_cm3': 0,
            'total_volume_cm3': 0,
            'avg_depth_mm': 0
        }
    
    total = len(potholes)
    severity_counts = {}
    total_volume = 0
    total_depth = 0
    
    for pothole in potholes:
        # Severity distribution
        severity = pothole.get('severity', 'Unknown')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Volume and depth stats
        volume = pothole.get('volumeCm3', 0)
        depth = pothole.get('dimensions', {}).get('max_depth', 0)
        
        total_volume += volume
        total_depth += depth
    
    avg_volume = total_volume / total if total > 0 else 0
    avg_depth = total_depth / total if total > 0 else 0
    
    return {
        'total': total,
        'severity_distribution': severity_counts,
        'avg_volume_cm3': round(avg_volume, 1),
        'total_volume_cm3': round(total_volume, 1),
        'avg_depth_mm': round(avg_depth, 1)
    }

def create_directories():
    """Create necessary directories"""
    directories = ['./data', './uploads', './3d_models', './processed']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_proccessed_depth_map(depth_map, camera_surface_distance=0.83, scale_factor = 0.01167):
    depth_map *= scale_factor
    proccessed_depth_map = depth_map - camera_surface_distance
    return proccessed_depth_map

# scale the depth map to get metric depth map, divide by the scale factor
# get the relative depth map, substract the depth map from the actual distace(camera height )
# return it