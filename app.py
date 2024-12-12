import os
import faiss
import torch
import numpy as np
from PIL import Image
from shutil import copyfile
from flask import Flask, render_template, request
from utils import get_model, get_transform, clear_folder

app = Flask(__name__)
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

model, device = get_model()
transform = get_transform()


index = faiss.read_index("dataset/index.faiss")
image_paths = np.load("dataset/images.npy", allow_pickle=True)



@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    clear_folder(app.config['UPLOAD_FOLDER'])
    clear_folder(app.config['RESULT_FOLDER'])

    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)


        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feature_vector = model(image_tensor).cpu().numpy()


        distances, indices = index.search(feature_vector, k=5)
        closest_images = [image_paths[idx] for idx in indices[0]]


        result_images = []
        for i, img_path in enumerate(closest_images):
            result_img_name = f"result_{i + 1}.jpg"
            result_img_path = os.path.join(app.config['RESULT_FOLDER'], result_img_name)
            copyfile(img_path, result_img_path)
            result_images.append(result_img_name)


        return render_template(
            'results.html', 
            uploaded_image=file.filename, 
            results=result_images
        )

if __name__ == '__main__':
    app.run(debug=True)