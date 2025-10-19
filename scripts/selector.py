# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Basil Benny
# Repository: github.com/basilbenny1002/Image-Selecter
# License notice: Permission is granted to use, copy, modify, and distribute this software
# for any purpose with or without fee, provided that the above notice appears in all copies.

#Necessary imports
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sqlite3
import numpy as np
import shutil
import os
from transformers import CLIPProcessor
from aesthetics_predictor import AestheticsPredictorV1



#Setting an environment variable to turn off warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#Load pretrained ResNet50 model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier layer
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Transform for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

#Creating database to store the mebeddings
conn = sqlite3.connect('embeddings.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS images
             (path TEXT PRIMARY KEY, embedding BLOB)''')
conn.commit()



#Load the aesthetics model
model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"
#Other models: https://huggingface.co/models?search=aesthetics-predictor-v1

predictor = AestheticsPredictorV1.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
predictor = predictor.to(device)


def add_image(image_path):
    """Add image embedding to DB.
    param image_path: Path to image file.
    """
    if not os.path.exists(image_path):
        print(f" File not found: {image_path}")
        return

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor).squeeze().numpy().astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)

    c.execute("INSERT OR REPLACE INTO images VALUES (?, ?)",
              (image_path, embedding.tobytes()))
    conn.commit()
    print(f" Added: {image_path}")



def find_and_remove_similar(image_path, similarity_threshold=0.8):
    """Return list of similar image paths and remove them from DB.
    param image_path: Path to query image file.
    param similarity_threshold: Cosine similarity threshold (0 to 1)."""
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        query_emb = model(tensor).squeeze().numpy().astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb)
    c.execute("SELECT path, embedding FROM images")
    all_data = c.fetchall()

    similar_paths = []
    for path, emb_blob in all_data:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        sim = np.dot(query_emb, emb)
        print(f"\n\n\n SIMILARITY THRESHOLD: {sim}\n\n\n")
        if sim >= similarity_threshold:
            similar_paths.append(path)

    # Remove found images from DB
    for path in similar_paths:
        c.execute("DELETE FROM images WHERE path=?", (path,))
    conn.commit()
    try:
        print(f"Found {len(similar_paths)} similar images (and removed them from DB).")
    except UnicodeEncodeError:
        print(f"Found {len(similar_paths)} similar images (and removed them from DB).")
    if image_path not in similar_paths:
        similar_paths.append(image_path)
    return similar_paths

def predict_image_aesthetic(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad(): 
        outputs = predictor(**inputs)
    prediction = outputs.logits

    score = prediction[0].item()
    return score


def choose(input_dir, output_dir, similarity):
    # input_dir = r"C:\Coding projects\VS Code\Image Selecter\Test_images" #Replace with the file having images to be processed
    # output_dir = r"C:\Coding projects\VS Code\Image Selecter\outputdir" #Replace with the file where you want to save the best images
    # similarity = 0.87 #Adjust similarity threshold as needed (0 to 1). Higher values would result in fewer images in the output and a lower value would mean more images. 0.87 is a good starting point.


    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        add_image(file_path)
    i=0

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        try:
            similar = find_and_remove_similar(file_path, similarity_threshold=similarity)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"An exception occurred:{e}")
            similar = []
        i+=1
        best_score = 0
        best_path = ""
        for path in similar:
            if not os.path.exists(rf"{input_dir}\{i}"):
                os.mkdir(rf"{input_dir}\{i}")
            scores = predict_image_aesthetic(path)
            if scores > best_score:
                best_score = scores
                best_path = path
            shutil.copy(path, rf"{input_dir}{i}")
        if best_path:
            shutil.copy(best_path, output_dir)
            os.remove(best_path)
        for path in similar:
            if not path == best_path:
                os.remove(path=path)


    
            



