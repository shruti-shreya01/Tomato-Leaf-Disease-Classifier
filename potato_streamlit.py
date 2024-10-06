



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





# import streamlit as st
# import tensorflow as tf
# from PIL import Image, ImageOps
# import numpy as np
# import pickle
# import os

# # Set the title of the app
# st.title("🌱 IIIT Lucknow - Potato Disease Prediction 🌱")
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

#         # Ensure the predictions are correctly normalized
#         predictions = tf.nn.softmax(predictions).numpy()  # Use softmax to normalize

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

# # Add a Rerun button to allow users to reset the app
# if st.button("🔄 Rerun"):
#     st.rerun()

# # Add some padding at the end of the app
# st.markdown("<br><br>", unsafe_allow_html=True)







import streamlit as st
import pickle
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageOps
import numpy as np

IMAGE_SIZE = 256

# Function to load and preprocess the uploaded image
def load_and_preprocess_image(image):
    img = load_img(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Path to the pickle file
file_path = "potato_pickle_final (1).pkl"

# Check if file exists and load the model
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Reconstruct the model from the architecture
    model = tf.keras.models.model_from_json(data["architecture"])
else:
    st.error(f"Model file not found at path: {file_path}")
    st.stop()

class_names = {0: "Early Blight", 1: "Late Blight", 2: "Healthy"}

# Initialize session state variables
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "confidence" not in st.session_state:
    st.session_state["confidence"] = None

# Enhanced UI
st.markdown("<h1 style='text-align: center; color: green;'>🍃 Potato Leaf Health Check 🍃</h1>", unsafe_allow_html=True)
st.write("Welcome farmers! Upload a photo of your potato leaf and let our AI help you diagnose its health.")

# File uploader for image input
uploaded_file = st.file_uploader("Select an image of a potato leaf...", type=["jpg", "jpeg", "png"], key="uploaded_file")

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    img_array = load_and_preprocess_image(uploaded_file)

    # Predict the class of the leaf disease
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100  # Confidence score in percentage

    # Store results in session state
    st.session_state["prediction"] = predicted_class
    st.session_state["confidence"] = confidence

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Map predicted class to the disease name
    disease_name = class_names.get(predicted_class, "Unknown")

    # Show prediction results
    st.markdown(f"### 🌿 Predicted Disease: **{disease_name}**")
    st.markdown(f"### 🔍 Confidence Score: **{confidence:.2f}%**")

    # Additional Tips for Farmers
    st.markdown("#### 🛠 Tips:")
    if predicted_class == 0:  # Early Blight
        st.write("⚠️ Early Blight detected. Consider using fungicides and practicing crop rotation.")
    elif predicted_class == 1:  # Late Blight
        st.write("⚠️ Late Blight detected. Immediate attention is required, use disease-resistant potato varieties.")
    elif predicted_class == 2:  # Healthy
        st.write("✅ Your leaf is healthy! Keep up the good farming practices.")

    # Optional: Display a pie chart with prediction probabilities
    st.write("### 🔢 Prediction Probabilities:")
    prob_df = {class_names[i]: float(prediction[0][i]) * 100 for i in range(len(class_names))}
    st.bar_chart(prob_df)
    if st.button("🔄 Rerun"):
        st.rerun()

# Sidebar enhancements
st.sidebar.title("About the Disease Classifier")
st.sidebar.info("This tool uses AI to detect common diseases in potato leaves. It's designed to help farmers identify potential issues early.")

st.sidebar.subheader("Disease Types")
st.sidebar.write("🌱 **Early Blight:** A common potato disease caused by a fungus.")
st.sidebar.write("🌱 **Late Blight:** A serious disease that can devastate potato crops.")
st.sidebar.write("🌱 **Healthy:** No signs of disease detected.")

st.sidebar.subheader("How It Works")
st.sidebar.write("Upload a clear image of your potato leaf, and our AI will predict its health based on trained models.")
