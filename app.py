from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image  # for image loading without cv2

app = Flask(__name__)

# ------------------------
# Load your trained model once
# ------------------------
MODEL_PATH = "DenseNetSVM_Model.h5"
model = load_model(MODEL_PATH)

# Classes (ensure same order used during training)
class_names = ["Healthy", "RedRot", "RedRust"]

# ------------------------
# Remedies (unchanged)
# ------------------------
remedies = {
    "RedRot": """ 
    Suggested remedies
    ✔ Remove and destroy affected canes to prevent spread.  
    ✔ Avoid waterlogging in fields.  
    ✔ Use resistant sugarcane varieties.  
    ✔ Apply fungicides like Carbendazim (0.1%) as recommended.  
    ✔ Ensure proper drainage and field sanitation.  
    ✔ Rotate crops to reduce pathogen buildup in soil.  

    🔗 Suggested Products:  
    - <a href="https://www.amazon.in/Bavistin-Carbendazim-Systemic-Fungicide-Protection/dp/B0CLGY1V4R" target="_blank">Crystal Bavistin Carbendazim 50% WP Systemic Fungicide</a>  
    - <a href="https://www.amazon.in/BAVISTIN-50-W-P-Carbendazim-250Gm/dp/B0FC6J8TMG" target="_blank">BAVISTIN 50% W.P (Carbendazim 50% WP) 250Gm</a>  
    - <a href="https://www.amazon.in/carbendazim-50-wp-1kg/s?k=carbendazim+50+wp+1kg" target="_blank">Rajshree WARDEN Carbendazim 50% WP 1kg</a>  
    - <a href="https://www.amazon.in/STUFF-CARBENDAZIM-MANCOZEB-250GM-Grams/dp/B0C3458B2S" target="_blank">HPM STUFF (Carbendazim 12% + Mancozeb 63%) 250GM</a>  
    - <a href="https://www.amazon.in/UPL-STANDUPSAAF-Carbendazinm12-Mancozeb63-Action/dp/B0BVZRM75D" target="_blank">UPL STANDUPSAAF (Carbendazim 12% | Mancozeb 63%)</a>  
    """,

    "RedRust": """     
    Suggested remedies
    ✔ Remove and burn infected leaves to reduce inoculum.  
    ✔ Improve air circulation by avoiding dense planting.  
    ✔ Spray fungicides such as Mancozeb (0.25%) or Propiconazole (0.1%).  
    ✔ Plant rust-resistant sugarcane varieties if available.  
    ✔ Maintain field hygiene to prevent spread.  
    ✔ Monitor regularly during the wet season when rust spreads faster.  

    🔗 Suggested Products:  
    - <a href="https://www.amazon.in/mancozeb-m45/s?k=mancozeb+m45" target="_blank">INDOFIL M-45 Mancozeb Fungicide 100gm</a>  
    - <a href="https://www.amazon.in/Master-Fungicide-Metalaxyl-Mancozeb-500gm/dp/B0CVQX25YT" target="_blank">Master (Metalaxyl 8% + Mancozeb 64% WP) 500gm Pack</a>  
    - <a href="https://www.amazon.in/mancozeb/s?k=mancozeb" target="_blank">Methox Zeb (Metalaxyl-M 4% + Mancozeb 64% WP) 250 gm</a>  
    - <a href="https://www.amazon.in/Carbendazim12-Mancozeb63-powdery-anthracnose-Alternaria/dp/B0DTGJ788B" target="_blank">MANCO CHAMP (Carbendazim 12% + Mancozeb 63%) 250 gm</a>  
    - <a href="https://www.amazon.in/fungicide-Systemic-Fungicide-metalaxyl-Mancozeb/dp/B0BSV6196K" target="_blank">Metmack Systemic and Contact Fungicide (100 gm)</a>  
    """,

    "Healthy": "✅ The leaf is healthy. No treatment required."
}

# ------------------------
# Prediction function using keras.preprocessing.image (no cv2)
# ------------------------
def model_predict(file_path):
    try:
        img = image.load_img(file_path, target_size=(224, 224))  # adjust size if needed
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)
        predicted_class = class_names[np.argmax(preds)]
        return predicted_class
    except Exception as e:
        print("[ERROR] Prediction failed:", e)
        return "Error"

# ------------------------
# Flask routes
# ------------------------
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            upload_folder = "static/uploads"
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            predicted_disease = model_predict(file_path)
            remedy = remedies.get(predicted_disease, "No remedy found.")

            return render_template("result.html",
                                   disease=predicted_disease,
                                   remedy=remedy,
                                   image_path=file_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
