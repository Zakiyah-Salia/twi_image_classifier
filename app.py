import os
import uuid
import gdown
import tempfile
import threading
import time
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory

app = Flask(__name__)

# Twi translations
twi_translations = {
    'Accidents and disaster': 'Asiane ne Amanehunu',
    'Agriculture': 'Kuadwuma',
    'Architecture': 'Adan mu nhyehyɛe',
    'Arts and crafts': 'Adwinneɛ ne nsaanodwuma',
    'Automobile': 'Kaa/ Kwan so nnwuma deɛ',
    'Construction': 'Adesie',
    'Culture': 'Amammerɛ',
    'Disabilities': 'Dɛmdi ahorow',
    'Economy': 'Sikasɛm ne Ahonya ho nsɛm',
    'Education': 'Nwomasua/Adesua',
    'Energy': 'Ahoɔden',
    'Engineering': 'Mfiridwuma',
    'Entertainment': 'Anigyedeɛ',
    'Ethnicity people and race': 'Mmusuakuw mu nnipa ne abusuakuw',
    'Family and society': 'Abusua ne Ɔmanfoɔ',
    'Fashion and clothing': 'Ahosiesie ne Ntadeɛ',
    'Fauna and flora': 'Mmoa ne Nnua',
    'Food and drink': 'Aduane ne Nsa',
    'Funeral': 'Ayie',
    'Furniture': 'Efie adeɛ / Efie hyehyeɛ',
    'Geography': 'Asase ho nimdeɛ',
    'Governance': 'Nniso nhyehyɛe',
    'Health and medicine': 'Apɔmuden ne Nnuro',
    'History': 'Abakɔsɛm',
    'Home and housing': 'Efie ne Tenabea',
    'Hospitality': 'Ahɔhoyɛ',
    'Immigration': 'Atubrafo ho nsɛm',
    'Justice and law enforcement': 'Atɛntenenee ne Mmara banbɔ',
    'Languages and Communication': 'Kasa ne Nkitahodie',
    'Leisure': 'Ahomegyeɛ',
    'Lifestyle': 'Abrateɛ',
    'Love and romance': 'Ɔdɔ ne Ɔdɔ ho nsɛm',
    'Marine': 'Ɛpo mu nsɛm',
    'Mining': 'Awuto fagude',
    'Movie cinema and theatre': 'Sinima ne Agorɔhwɛbea',
    'Music and dance': 'Nnwom ne Asaw',
    'Nature': 'Abɔdeɛ',
    'News': 'Kaseɛbɔ',
    'Politics': 'Amammuisɛm',
    'Religion': 'Gyidi ne Nsom',
    'Sanitation': 'Ahoteɛ',
    'Science': 'Saense',
    'Security': 'Banbɔ',
    'Sports': 'Agodie',
    'Technology': 'Tɛknɔlɔgyi',
    'Trading and commerce': 'Dwadie ne Nsesaguoɔ',
    'Transportation': 'Akwantuo',
    'Travel and tourism': 'Akwantuɔ ne Ahɔhoɔ',
    'Weather and climate': 'Ewiem tebea ne Ewiem nhyehyɛeɛ'
}

class_names = list(twi_translations.keys())

MODEL_PATH = "fine_tuned_model_3.0.keras"
FILE_ID = "1gdfeYaa8X66J4IG9t830DKUK5LcIqfJO"

def download_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

download_model()
model = load_model(MODEL_PATH)

# Auto-delete function
def delete_later(path, delay=300):
    def delete():
        time.sleep(delay)
        try:
            os.remove(path)
            print(f"Deleted: {path}")
        except FileNotFoundError:
            pass
    threading.Thread(target=delete).start()

# Image preprocessing
def process_uploaded_image(uploaded_file, img_size=(224, 224)):
    with tempfile.TemporaryDirectory() as tmpdir:
        class_dir = os.path.join(tmpdir, "class0")
        os.makedirs(class_dir, exist_ok=True)

        image_path = os.path.join(class_dir, uploaded_file.filename)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        dataset = image_dataset_from_directory(
            tmpdir,
            image_size=img_size,
            batch_size=1,
            shuffle=False
        )

        for images, _ in dataset.take(1):
            return images

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files or request.files["image"].filename == "":
            return render_template("index.html", predictions=None, warning="Please upload an image.")

        image_file = request.files["image"]
        image_file.seek(0)
        img_tensor = process_uploaded_image(image_file)

        preds = model.predict(img_tensor)[0]
        top_indices = preds.argsort()[::-1][:3]

        top_preds = []
        for i in top_indices:
            eng_label = class_names[i]
            twi_label = twi_translations.get(eng_label, "❓")
            confidence = float(preds[i])
            top_preds.append({
                "english": eng_label,
                "twi": twi_label,
                "confidence": confidence
            })

        if top_preds[0]["confidence"] < 0.5:
            return render_template("index.html", predictions=[], warning="Gyidie no nsɔ 50%, enti yɛrentumi nka")

        # Save image for preview
        filename = f"{uuid.uuid4().hex}_{image_file.filename}"
        static_path = os.path.join("static", filename)
        image_file.seek(0)
        with open(static_path, "wb") as f:
            f.write(image_file.read())

        # Schedule deletion
        delete_later(static_path, delay=300)  # Delete after 5 minutes

        image_url = url_for('static', filename=filename)
        return render_template("index.html", predictions=top_preds, warning=None, image_url=image_url)

    return render_template("index.html", predictions=None, warning=None)

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)
