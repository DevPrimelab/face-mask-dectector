from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import os

# Flask app start
app = Flask(__name__)

# Model load
model = tf.keras.models.load_model("mask_model.h5")

# Upload folder
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""

    if request.method == 'POST':
        file = request.files['file']

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Image process
            img = image.load_img(filepath, target_size=(150,150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = model.predict(img_array)

            if prediction[0][0] > 0.5:
                result = "❌ Mask nahi pehna"
            else:
                result = "😷 Mask pehna hai"

    return render_template('index.html', result=result)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
