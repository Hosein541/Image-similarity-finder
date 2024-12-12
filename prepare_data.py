import os
import torch
import faiss
import numpy as np
from torchvision.datasets import CIFAR10
from utils import get_model, get_transform

os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
os.makedirs("dataset", exist_ok=True)
os.makedirs("dataset/images", exist_ok=True)


model, device = get_model()
transform_feature = get_transform()

dataset = CIFAR10(root="dataset", train=True, download=True)
subset = torch.utils.data.Subset(dataset, range(3000))
os.remove("dataset/cifar-10-python.tar.gz")


def extract_features():
    vectors = []
    image_paths = []

    for idx, (img, _) in enumerate(subset):
        try:
            image_path = f"dataset/images/image_{idx}.jpg"
            img.save(image_path)
            image_paths.append(image_path)

            img_tensor = transform_feature(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feature_vector = model(img_tensor).cpu().numpy()
            vectors.append(feature_vector)

        except Exception as e:
            print(f"Error processing image {idx}: {e}")

    vectors = np.vstack(vectors).astype('float32')
    np.save("dataset/vectors.npy", vectors)
    np.save("dataset/images.npy", np.array(image_paths, dtype=object))

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, "dataset/index.faiss")
    print("FAISS index saved!")


if not os.path.exists("dataset/vectors.npy") or not os.path.exists("dataset/images.npy"):
    print("Extracting features...")
    extract_features()
    print("Features saved!")
else:
    print("Features already exist.")
