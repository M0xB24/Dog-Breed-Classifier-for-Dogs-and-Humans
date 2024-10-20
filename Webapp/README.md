# Dog Breed Classifier for Dogs and Humans

### Instructions:

1. Adjust the path in the run.py load_resources() method for the Xception_model and the list of dog breed names.
   The files are also included in the repo.
    1.1. Xception_model = load_model("D:/GIT/Assignement7/Webapp/models/Xception.keras")
    1.2. dog_names = [item[20:-1] for item in sorted(glob("D:/GIT/Assignement7/dog_images_complete/dogImages/train/*/"))]

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/
