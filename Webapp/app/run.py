import cv2
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from extract_bottleneck_features import *
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image                  
from keras.applications.resnet50 import ResNet50
from mtcnn import MTCNN       
import numpy as np
from glob import glob
from tensorflow.keras.models import load_model
import csv

# Initialisiere Flask-Anwendung und Upload-Ordner, Setze den absoluten Pfad für den Upload-Ordner
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Globale Variablen für die Modelle und die Hunderassenliste
Xception_model = None
dog_names = []
ResNet50_model = None

#---------------
# UPLOAD PICTURE
#---------------

# Funktion für Bild-Upload und Speichern
def upload_data(file):
    """
    Function to process selected image in Webapp. 
    The image is uploaded to a static/uploads in the same path as the run.py is placed.
    """
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    filename = secure_filename(file.filename)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(img_path)
    return img_path


#---------------
# HELPER METHODS
#---------------

def path_to_tensor(img_path):
    """
    Helper function to convert an image from a numpy array to a 4D-tensor with shape (nb_samples,224,224,3).
    """
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        return np.expand_dims(x, axis=0)
    except IOError:
        print(f"Warning: Skipping corrupted image {img_path}")
        return None

def clean_breed_name(prediction):
    """
    Helper function To clean the string of dog breed names. 
    The original string is a path. The function splits it at the last point from the right side and replaces _ by " ".
    """
    breed_name = prediction.rsplit(".", 1)[-1]
    return breed_name.replace("_", " ")

#Lädt das Xception-Modell, die Hunderassenliste und das resnet50 nur einmal beim Start um die Performance zu verbessern
def load_resources():
    """
    Load the required models for the human identification, as well as the dog identification and the dog breed classification.
    The models are only loaded once to have a better performance within the web app. 
    """
    global Xception_model, dog_names, ResNet50_model
    if Xception_model is None or dog_names is None or ResNet50_model is None:
        # Lade das Xception-Modell mit einem relativen Pfad
        model_path = os.path.join(BASE_DIR, "models_data", "Xception.keras")
        Xception_model = load_model(model_path)
        
        # Lade die Hunderassen-Namen <--- Das hier noch in einer csv speicher!
        dog_images_path = os.path.join(BASE_DIR, "models_data", "dog_names.csv")
        # Einlesen der CSV-Datei und Speichern der Pfade in einer Liste
        with open(dog_images_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
            # Jede 'row' enthält eine Liste mit einem einzelnen Pfad
                dog_names.append(row[0])
        
        # Lade das ResNet50-Modell (vorgefertigte Gewichte von ImageNet)
        ResNet50_model = ResNet50(weights='imagenet')

#-----------------
# HUMAN PREDICTION
#-----------------
def face_detector(image_path):
    """
    The function gets a path to and image and identifies with the help of the trained MTCN model if it is a human face. 
    If so it returns True otherwise False.
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(image)
    return len(faces) > 0


#---------------
# DOG PREDICTION
#---------------
def ResNet50_predict_labels(img_path):
    """
    The function gets a path to and image and identifies with the help of the RestNet50 model if it is a dog. 
    The function returns an array whose 𝑖-th entry is the model's predicted probability that the image belongs 
    to the  𝑖-th ImageNet category. 
    """    
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    """
    The function gets a path to and image and identifies with the help of the trained RestNet50 model if it is a dog. 
    Therefore it calls the method ResNet50_predict_labels() which returns values for dogs between 151-268. 
    If The value is in between this range then the function returns True otherwise False.
    """    
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


#-------------------------
# DOG BREED CLASSIFICATION
#-------------------------
def predict_dog_breed(img_path):
    """
    The function gets a path to and image and classifies with the help of the trained extract_Xception model which dog breed it is. 
    The method returns the classification result which is a name from a list of 133 dog breed names.
    """   
    load_resources() #Lädt Modelle
    bottleneck_features = extract_Xception(path_to_tensor(img_path))
    predicted_vector = Xception_model.predict(bottleneck_features)
    breed_index = np.argmax(predicted_vector)
    breed_name = clean_breed_name(dog_names[breed_index])
    return breed_name

def predict_image_category(img_path):
    """
    The function uses the dog_detector, face_detector and predict_dog_breed function to process an image.
    The input to this function is a img_path.
    The outpur of this function is wheter it's dog, a human, the dog breed.
    """  
    if dog_detector(img_path):
        dog_breed = predict_dog_breed(img_path)
        return True, False, dog_breed
    elif face_detector(img_path):
        dog_breed = predict_dog_breed(img_path)
        return False, True, dog_breed
    else:
        return False, False, "Weder ein Hund noch ein Mensch wurde erkannt. Bitte geben Sie ein anderes Bild ein."


@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')

@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload():
    """
    The function handles the flask upload of the image in the web app. 
    It check's if in the HTTP-Post request contains a file.
    It check's if an image was selected pre to uplaoding.
    It stores the uplaoded image with the help of upload_data if it's a valid file.
    It creates a relative path for the web app to the uplaoded image.
    The method for identification and classification of human, dog and breed is called.
    The go html get's the results of identification and classification is rendered.
    """  
    if 'file' not in request.files:
        print("No file part in the request")
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        print("No file selected for uploading")
        return redirect(request.url)

    if file:
        img_path = upload_data(file)

        # Erzeuge den relativen Pfad zu `static/uploads` für die Anzeige im Browser
        img_url = url_for('static', filename=f'uploads/{os.path.basename(img_path)}')
        
        is_dog, is_human, dog_breed = predict_image_category(img_path)

        return render_template(
            'go.html',
            img_url=img_url,
            is_dog=is_dog,
            is_human=is_human,
            dog_breed=dog_breed
        )

def main():
    """
    Main Method to load the required resoureces once and start the wab app.
    """  
    load_resources()  # Lade alle Ressourcen beim Start der Anwendung
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
