import numpy as np
from model import interpreter, input_details, output_details, CLASS_MAPPING

def predict_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return {
        "disease": CLASS_MAPPING.get(class_index, "Unknown"),
        "confidence": confidence,
        "model_version": "tflite_v1"
    }