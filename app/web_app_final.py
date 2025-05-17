import os
import zipfile
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from calculate_metrics import calculate_metrics
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from transformers import ViTForImageClassification, ViTImageProcessor
import torch

app = Flask(__name__)

# Configure upload folders
UPLOAD_FOLDER = 'uploads'
SINGLE_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'single')
MULTI_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'multi')
OUTPUT_DIR = 'results'

os.makedirs(SINGLE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MULTI_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'zip'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SINGLE_UPLOAD_FOLDER'] = SINGLE_UPLOAD_FOLDER
app.config['MULTI_UPLOAD_FOLDER'] = MULTI_UPLOAD_FOLDER

# Load the segmentation model
segmentation_model = load_model('../outputs/unet/unet_model.h5', compile=False)

# Load the transformer classification model
classification_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


# Function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def classify_image(image_path):
    """Classify wound image using the transformer model."""
    image = Image.open(image_path).convert('RGB')
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = classification_model(**inputs)
    logits = outputs.logits 
    predicted_class = torch.argmax(logits, dim=1).item()
    class_labels = classification_model.config.id2label
    return class_labels[predicted_class]


def generate_outputs(image_path, output_folder):
    # Load and resize the original image
    original_image = Image.open(image_path).convert('RGB')
    original_image_resized = original_image.resize((256, 256))
    original_image_array = img_to_array(original_image_resized) / 255.0

    # Predict segmentation mask
    pred_mask = segmentation_model.predict(np.expand_dims(original_image_array, axis=0))[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    # Save outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save predicted mask
    output_mask_path = os.path.join(output_folder, f'{base_name}_predicted_mask.png')
    pred_mask_resized = cv2.resize(pred_mask, (original_image.size[0], original_image.size[1]))
    cv2.imwrite(output_mask_path, pred_mask_resized * 255)

    # Convert the original image from RGB to BGR
    original_image_bgr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    pred_mask_bgr = np.repeat(pred_mask_resized[:, :, np.newaxis], 3, axis=2) * 255

    # Create translucent overlays
    red_overlay = np.zeros_like(original_image_bgr, dtype=np.uint8)
    blue_overlay = np.zeros_like(original_image_bgr, dtype=np.uint8)
    red_overlay[:, :, 2] = 255
    blue_overlay[:, :, 0] = 255

    color_overlay = np.where(pred_mask_resized[:, :, np.newaxis], 
                             cv2.addWeighted(original_image_bgr, 1, red_overlay, 0.5, 0),
                             cv2.addWeighted(original_image_bgr, 1, blue_overlay, 0.5, 0))

    output_overlay_path = os.path.join(output_folder, f'{base_name}_color_overlay.png')
    cv2.imwrite(output_overlay_path, color_overlay)

    # Classify the image
    wound_type = classify_image(image_path)

    # Calculate metrics
    length, width, area, perimeter, circularity = calculate_metrics(pred_mask_resized)

    # Create comparison image
    comparison_image = np.hstack((original_image_bgr, pred_mask_bgr, color_overlay))
    blank_space = Image.new("RGB", (comparison_image.shape[1], 50), (0, 0, 0))
    draw = ImageDraw.Draw(blank_space)
    metrics_text = (f"Length: {length:.2f} mm | Width: {width:.2f} mm | Area: {area:.2f} mm² | "
                    f"Perimeter: {perimeter:.2f} mm | Circularity: {circularity:.2f} | "
                    f"Type: {wound_type}")

    font_path = "arial.ttf"
    font = ImageFont.truetype(font_path, size=10)
    draw.text((10, 10), metrics_text, font=font, fill=(255, 255, 255))

    comparison_with_metrics = np.vstack((comparison_image, np.array(blank_space)))
    output_comparison_path = os.path.join(output_folder, f'{base_name}_comparison.png')
    cv2.imwrite(output_comparison_path, comparison_with_metrics)

    return output_mask_path, output_overlay_path, output_comparison_path, wound_type


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        upload_type = request.form.get('upload_type')
        file = request.files.get('file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            if upload_type == 'multi' and filename.endswith('.zip'):
                file_path = os.path.join(app.config['MULTI_UPLOAD_FOLDER'], filename)
                file.save(file_path)

                extracted_folder = os.path.splitext(filename)[0]
                extraction_path = os.path.join(app.config['MULTI_UPLOAD_FOLDER'], extracted_folder)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extraction_path)
                os.remove(file_path)

                output_folder = os.path.join(OUTPUT_DIR, 'multi', extracted_folder)
                os.makedirs(output_folder, exist_ok=True)

                image_paths = [os.path.join(extraction_path, f) for f in os.listdir(extraction_path) if f.endswith(('png', 'jpg', 'jpeg'))]

                output_paths = []
                for image_path in image_paths:
                    output_paths.append(generate_outputs(image_path, output_folder))

                return render_template('results.html', message='Multiple images processed successfully!', outputs=output_paths, folder_name=extracted_folder, folder_type='multi')

            elif upload_type == 'single':
                file_path = os.path.join(app.config['SINGLE_UPLOAD_FOLDER'], filename)
                file.save(file_path)

                picture_name = os.path.splitext(filename)[0]
                single_output_folder = os.path.join(OUTPUT_DIR, 'single', picture_name)
                os.makedirs(single_output_folder, exist_ok=True)

                output_paths = generate_outputs(file_path, single_output_folder)

                return render_template('results.html', message='Single image processed successfully!', outputs=[output_paths], folder_name=picture_name, folder_type='single')


@app.route('/results/<folder_name>/<path:filename>', methods=['GET'])
def serve_image(folder_name, filename):
    file_path = os.path.join(OUTPUT_DIR, 'single', folder_name, filename.replace('\\', '/'))
    return send_file(file_path)


@app.route('/download/<folder_type>/<folder_name>', methods=['GET'])
def download(folder_type, folder_name):
    folder_path = os.path.join(OUTPUT_DIR, folder_type, folder_name)
    zip_filename = f"{folder_name}_results.zip"
    zip_filepath = os.path.join(OUTPUT_DIR, zip_filename)

    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

    return send_file(zip_filepath, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
