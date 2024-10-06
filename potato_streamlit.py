# import streamlit as st
# import tensorflow as tf
# from PIL import Image, ImageOps
# import numpy as np
# import pickle

# # Set the title of the app
# st.title("IIIT Lucknow - Potato Disease Prediction")
# st.write("""
#     Upload an image of a potato leaf, and the model will predict the disease.
# """)

# # Load the model from pickle file
# @st.cache_resource
# def load_model_from_pickle(pickle_path):
#     with open(pickle_path, "rb") as f:
#         model = pickle.load(f)
#     # If the model was saved without compilation, compile it here
#     if not hasattr(model, "optimizer"):
#         model.compile(
#             optimizer='adam',
#             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#             metrics=['accuracy']
#         )
#     return model

# # import pickle

# # # Path to the pickle file
# # model_path = r"C:\Users\shrey\Downloads\potato_pickle_final (1).pkl"

# # # Load the model
# # with open(model_path, 'rb') as f:
# #     model = pickle.load(f)

# import os

# # Path to the pickle file
# model_path = "potato_pickle_final (1).pkl"

# # Check if the file exists
# if not os.path.exists(model_path):
#     st.error(f"File not found: {model_path}")
# else:
#     # Load the model
#     with open(model_path, 'rb') as f:
#         model = pickle.load(f)


# # Define class names (modify these based on your dataset)
# class_names = ['Healthy', 'Early Blight', 'Late Blight', 'Leaf Curl', 'Other Diseases']

# # Function to preprocess the image
# # def preprocess_image(image: Image.Image) -> np.ndarray:
# #     IMAGE_SIZE = 256  # Must match the image size used during training
# #     image = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
# #     image_array = np.asarray(image)
    
# #     # Convert RGBA to RGB if necessary
# #     if image_array.shape[-1] == 4:
# #         image_array = image_array[..., :3]
    
# #     image_array = image_array / 255.0  # Rescale to [0,1]
# #     image_array = np.expand_dims(image_array, axis=0)  # Create batch axis
# #     return image_array

# from PIL import Image, ImageOps
# import numpy as np

# # Function to preprocess the image
# def preprocess_image(image: Image.Image) -> np.ndarray:
#     IMAGE_SIZE = 256  # Must match the image size used during training
#     image = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
#     image_array = np.asarray(image)

#     # Convert RGBA to RGB if necessary
#     if image_array.shape[-1] == 4:
#         image_array = image_array[..., :3]

#     image_array = image_array / 255.0  # Rescale to [0, 1]
#     image_array = np.expand_dims(image_array, axis=0)  # Create batch axis
#     return image_array


# # File uploader allows users to upload images
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     try:
#         # Display the uploaded image
#         image = Image.open(uploaded_file).convert('RGB')
#         st.image(image, caption='Uploaded Image.', use_column_width=True)
#         st.write("Classifying...")

#         # Preprocess the image
#         processed_image = preprocess_image(image)

#         # Make prediction
#         predictions = model.predict(processed_image)

#         # If the model outputs probabilities, ensure they sum to 1
#         if not np.isclose(predictions.sum(), 1):
#             predictions = tf.nn.softmax(predictions).numpy()

#         confidence = np.max(predictions) * 100
#         predicted_class = class_names[np.argmax(predictions)]

#         # Display prediction
#         st.write(f"*Predicted Class:* {predicted_class}")
#         st.write(f"*Confidence:* {confidence:.2f}%")

#         # Optional: Display a bar chart of all class probabilities
#         st.write("### Prediction Probabilities")
#         prob_df = {class_names[i]: float(predictions[0][i]) * 100 for i in range(len(class_names))}
#         st.bar_chart(prob_df)
    
#     except Exception as e:
#         st.error(f"An error occurred during prediction: {e}")






import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pickle
import os

# Set the title of the app
st.title("IIIT Lucknow - Potato Disease Prediction")
st.write("""
    Upload an image of a potato leaf, and the model will predict the disease.
""")

# Path to the pickle file
model_path = "potato_pickle_final (1).pkl"

# Check if the file exists
if not os.path.exists(model_path):
    st.error(f"File not found: {model_path}")
else:
    try:
        # Load the model from the pickle file
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)  # Load the model as a dictionary

        # Check if the dictionary contains the model and its weights
        if 'model' in model_dict and 'model_weights.weights.h5' in model_dict:
            model = model_dict['model']  # Access the model from the dictionary
            model.load_weights(model_dict['model_weights.weights.h5'])  # Load weights
        else:
            st.error("Model or weights not found in the pickle file. Please check the saved model.")

        # If the model was saved without compilation, compile it here
        if not hasattr(model, "optimizer"):
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy']
            )

    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")

# Define class names (modify these based on your dataset)
class_names = ['Healthy', 'Early Blight', 'Late Blight']

# Function to preprocess the image
def preprocess_image(image: Image.Image) -> np.ndarray:
    IMAGE_SIZE = 256  # Must match the image size used during training
    image = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    image_array = np.asarray(image)

    # Convert RGBA to RGB if necessary
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]

    image_array = image_array / 255.0  # Rescale to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Create batch axis
    return image_array

# File uploader allows users to upload images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        predictions = model.predict(processed_image)

        # If the model outputs probabilities, ensure they sum to 1
        if not np.isclose(predictions.sum(), 1):
            predictions = tf.nn.softmax(predictions).numpy()

        confidence = np.max(predictions) * 100
        predicted_class = class_names[np.argmax(predictions)]

        # Display prediction
        st.write(f"*Predicted Class:* {predicted_class}")
        st.write(f"*Confidence:* {confidence:.2f}%")

        # Optional: Display a bar chart of all class probabilities
        st.write("### Prediction Probabilities")
        prob_df = {class_names[i]: float(predictions[0][i]) * 100 for i in range(len(class_names))}
        st.bar_chart(prob_df)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

