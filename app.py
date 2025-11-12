# ✅ FULL UPDATED app.py WITH HISTORY SUPPORT
import os
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from PIL import Image
import json
from datetime import datetime

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "saved_model" / "butterfly_model.h5"
LABELS_PATH = BASE_DIR / "saved_model" / "labels.json"
CSV_PATH = BASE_DIR / "species_information.csv"  # ✅ Updated filename
HISTORY_CSV = BASE_DIR / "history.csv"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
IMG_SIZE = 224
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# ✅ Load model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place your model there.")
model = load_model(str(MODEL_PATH))

# ✅ Load labels
labels_map = {}
if LABELS_PATH.exists():
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels_raw = json.load(f)
        if all(str(k).isdigit() for k in labels_raw.keys()):
            labels_map = {int(k): v for k, v in labels_raw.items()}
        else:
            labels_map = {int(v): k for k, v in labels_raw.items()}
        labels_map = {k: labels_map[k] for k in sorted(labels_map)}
    except Exception:
        labels_map = {}

# ✅ Fallback: infer classes from "train" folder
TRAIN_FOLDER = BASE_DIR / "train"
if not labels_map and TRAIN_FOLDER.exists():
    folders = sorted([d.name for d in TRAIN_FOLDER.iterdir() if d.is_dir()])
    labels_map = {i: folders[i] for i in range(len(folders))}

# ✅ Load species CSV
def load_species_csv():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, encoding="latin1")
        if "Species Name" in df.columns:
            df["__key__"] = df["Species Name"].astype(str).str.strip().str.upper()
        else:
            df["__key__"] = df.iloc[:,0].astype(str).str.strip().str.upper()
        return df
    else:
        cols = ["Species Name","Scientific Name","Life Span","Exotic or Not","Habitat/Location","Extra Info"]
        df = pd.DataFrame(columns=cols)
        df["__key__"] = ""
        return df

df_info = load_species_csv()

# ✅ Helpers
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=IMG_SIZE):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size, target_size))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_top1(image_path):
    arr = preprocess_image(image_path)
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    prob = float(preds[idx])
    name = labels_map.get(idx, str(idx))
    return name, prob

# ✅ Routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/history")
def history():
    if HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV)
        return render_template("history.html", data=df.to_dict(orient="records"))
    return render_template("history.html", data=[])

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = UPLOAD_FOLDER / filename
            file.save(save_path)

            try:
                species_name, prob = predict_top1(str(save_path))
            except Exception as e:
                return render_template("predict.html", error=str(e))

            # ✅ Save prediction to history.csv
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record = {
                "Image Name": filename,
                "Predicted Species": species_name,
                "Accuracy": f"{prob * 100:.2f}%",
                "Timestamp": timestamp
            }

            cols = ["Image Name", "Predicted Species", "Accuracy", "Timestamp"]
            if HISTORY_CSV.exists():
                df_old = pd.read_csv(HISTORY_CSV)
                df_new = pd.DataFrame([record])
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                df_combined.to_csv(HISTORY_CSV, index=False)
            else:
                df_new = pd.DataFrame([record], columns=cols)
                df_new.to_csv(HISTORY_CSV, index=False)

            # ✅ Match CSV info
            key = species_name.strip().upper()
            row = df_info[df_info["__key__"] == key]
            species_info = None
            if row.shape[0] > 0:
                r = row.iloc[0]
                species_info = {
                    "Species Name": r.get("Species Name", ""),
                    "Scientific Name": r.get("Scientific Name", ""),
                    "Life Span": r.get("Life Span", ""),
                    "Exotic or Not": r.get("Exotic or Not", r.get("Exotic or Native", "")),
                    "Habitat/Location": r.get("Habitat/Location", ""),
                    "Extra Info": r.get("Extra Info", "")
                }

            return render_template(
                "predict.html",
                uploaded_image=url_for("static", filename=f"uploads/{filename}"),
                species=species_name,
                accuracy=f"{prob*100:.2f}",
                species_info=species_info
            )

        else:
            return render_template("predict.html", error="Invalid file type. Use png/jpg/jpeg.")
    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)
