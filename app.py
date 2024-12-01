from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os
from src.models.classifier import GarbageClassifier
from src.config import Config
import logging
import numpy as np

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize the classifier
config = Config()
classifier = GarbageClassifier(config)

def get_waste_category(label):
    """Determine if waste is recyclable and which bin to use."""
    recyclable_items = {'cardboard', 'glass', 'metal', 'paper', 'plastic'}
    non_recyclable = {'trash'}
    
    if label in recyclable_items:
        return {
            'is_recyclable': True,
            'bin_color': 'blue',
            'message': f'This {label} item is recyclable and should go in the BLUE recycling bin.'
        }
    elif label in non_recyclable:
        return {
            'is_recyclable': False,
            'bin_color': 'gray',
            'message': f'This {label} item is not recyclable and should go in the GRAY trash bin.'
        }
    return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image was sent
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image uploaded',
                'message': 'Please upload an image'
            }), 400

        # Read image file
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Save image temporarily
        temp_path = 'temp_image.jpg'
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)

        # Get prediction
        results = classifier.model.predict(
            source=temp_path,
            conf=0.25,
            save=False
        )

        # Remove temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Process results
        result = results[0]
        boxes = result.boxes

        if len(boxes) > 0:
            # Get the prediction with highest confidence
            best_box = boxes[0]
            confidence = float(best_box.conf[0])
            class_id = int(best_box.cls[0])
            predicted_label = config.CLASS_NAMES[class_id]

            # Get waste category information
            category_info = get_waste_category(predicted_label)

            return jsonify({
                'success': True,
                'label': predicted_label,
                'confidence': float(confidence),
                'is_recyclable': category_info['is_recyclable'],
                'bin_color': category_info['bin_color'],
                'message': category_info['message']
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No objects detected. Please upload another image.'
            })

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({
            'error': 'Error processing image',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
