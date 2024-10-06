



# import streamlit as st
# import tensorflow as tf
# from PIL import Image, ImageOps
# import numpy as np
# import pickle
# import os

# # Set the title of the app
# st.title("Potato Disease Prediction")
# st.write("""
#     Upload an image of a potato leaf, and the model will predict the disease.
# """)

# # Load the model from pickle file
# @st.cache_resource
# def load_model_from_pickle(pickle_path):
#     # Check if the file exists before trying to load it
#     if not os.path.exists(pickle_path):
#         st.error(f"Model file not found at path: {pickle_path}")
#         st.stop()

#     with open(pickle_path, "rb") as f:
#         data = pickle.load(f)

#     # Reconstruct the model from the architecture
#     model = tf.keras.models.model_from_json(data["architecture"])

#     # Load the weights from the file
#     model.load_weights(data["weights"])

#     # Compile the model if needed
#     model.compile(
#         optimizer='adam',
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#         metrics=['accuracy']
#     )
#     return model

# # Load the pre-trained model (make sure the file exists in the working directory)
# model = load_model_from_pickle("potato_pickle_final (1).pkl")

# # Define class names (modify these based on your dataset)
# class_names = ['Healthy', 'Early Blight', 'Late Blight']

# # Function to preprocess the image
# def preprocess_image(image: Image.Image) -> np.ndarray:
#     IMAGE_SIZE = 256  # Must match the image size used during training
#     image = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
#     image_array = np.asarray(image)
    
#     # Convert RGBA to RGB if necessary
#     if image_array.shape[-1] == 4:
#         image_array = image_array[..., :3]
    
#     image_array = image_array / 255.0  # Rescale to [0,1]
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
st.title("🌱 IIIT Lucknow - Potato Disease Prediction 🌱")
st.write("""
    Upload an image of a potato leaf, and the model will predict the disease.
""")

# Load the model from pickle file
@st.cache_resource
def load_model_from_pickle(pickle_path):
    # Check if the file exists before trying to load it
    if not os.path.exists(pickle_path):
        st.error(f"Model file not found at path: {pickle_path}")
        st.stop()

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    # Reconstruct the model from the architecture
    model = tf.keras.models.model_from_json(data["architecture"])
    
    # Load the weights from the file
    model.load_weights(data["model_weights.weights.h5"])

    # Compile the model if needed
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model

# Load the pre-trained model (make sure the file exists in the working directory)
model = load_model_from_pickle("potato_pickle_final (1).pkl")

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
    
    image_array = image_array / 255.0  # Rescale to [0,1]
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

        # Ensure the predictions are correctly normalized
        predictions = tf.nn.softmax(predictions).numpy()  # Use softmax to normalize

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

# Add a Rerun button to allow users to reset the app
if st.button("🔄 Rerun"):
    st.rerun()

# Add some padding at the end of the app
st.markdown("<br><br>", unsafe_allow_html=True)
