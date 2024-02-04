import cv2
import time
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

data=np.ndarray(shape=(1,224,224,3),dtype=np.float32)
# Open a connection to the webcam (usually webcam index 0)
cap = cv2.VideoCapture(0)
# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
# Read and display frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Check if the frame is read successfully
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Preprocess the frame (resize, normalize, etc.) as needed for your neural network
    processed_frame = cv2.resize(frame, (224, 224)) # Example: Resize to 224x224
    # Convert the frame to a NumPy array (raw image data)
    raw_image_data = np.array(processed_frame)
    normalized_image_array = (raw_image_data.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

# Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    print(class_name[2:7])
    b=''.join(class_name[2:7])

    if confidence_score>0.51 and b == "Human":
  #np.set_printoptions(suppress=True)

# Load the model
        model1 = load_model("keras_model1.h5", compile=False)
        model1.summary()
# Load the labels
        class_names1 = open("labels1.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
  #data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
  #image = Image.open("IMG_1243.jpeg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
  #size = (224, 224)
  #image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
  #image_array = np.asarray(image)

# Normalize the image
  #normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
  #data[0] = normalized_image_array

# Predicts the model
        prediction = model1.predict(data)
        index = np.argmax(prediction)
        class_name = class_names1[index]
        confidence_score = prediction[0][index]

# Print prediction and confidence score
        print("hi")
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)

       
    # Now, you can use 'raw_image_data' as input to your neural network

    # Display the frame
        cv2.imshow('Webcam', frame)
    # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
# Release the webcam and close all OpenCV windows
    
import cv2
import time
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

data=np.ndarray(shape=(1,224,224,3),dtype=np.float32)
# Open a connection to the webcam (usually webcam index 0)
cap = cv2.VideoCapture(0)
# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
# Read and display frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Check if the frame is read successfully
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Preprocess the frame (resize, normalize, etc.) as needed for your neural network
    processed_frame = cv2.resize(frame, (224, 224)) # Example: Resize to 224x224
    # Convert the frame to a NumPy array (raw image data)
    raw_image_data = np.array(processed_frame)
    normalized_image_array = (raw_image_data.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

# Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    print(class_name[2:7])
    b=''.join(class_name[2:7])

    if confidence_score>0.51 and b == "Human":
  #np.set_printoptions(suppress=True)

# Load the model
        model1 = load_model("keras_model1.h5", compile=False)
        model1.summary()
# Load the labels
        class_names1 = open("labels1.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
  #data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
  #image = Image.open("IMG_1243.jpeg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
  #size = (224, 224)
  #image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
  #image_array = np.asarray(image)

# Normalize the image
  #normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
  #data[0] = normalized_image_array

# Predicts the model
        prediction = model1.predict(data)
        index = np.argmax(prediction)
        class_name = class_names1[index]
        confidence_score = prediction[0][index]

# Print prediction and confidence score
        print("hi")
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)

       
    # Now, you can use 'raw_image_data' as input to your neural network

    # Display the frame
        cv2.imshow('Webcam', frame)
    # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
# Release the webcam and close all OpenCV windows
    
import cv2
import time
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

data=np.ndarray(shape=(1,224,224,3),dtype=np.float32)
# Open a connection to the webcam (usually webcam index 0)
cap = cv2.VideoCapture(0)
# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
# Read and display frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Check if the frame is read successfully
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Preprocess the frame (resize, normalize, etc.) as needed for your neural network
    processed_frame = cv2.resize(frame, (224, 224)) # Example: Resize to 224x224
    # Convert the frame to a NumPy array (raw image data)
    raw_image_data = np.array(processed_frame)
    normalized_image_array = (raw_image_data.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

# Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    print(class_name[2:7])
    b=''.join(class_name[2:7])

    if confidence_score>0.51 and b == "Human":
  #np.set_printoptions(suppress=True)

# Load the model
        model1 = load_model("keras_model1.h5", compile=False)
        model1.summary()
# Load the labels
        class_names1 = open("labels1.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
  #data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
  #image = Image.open("IMG_1243.jpeg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
  #size = (224, 224)
  #image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
  #image_array = np.asarray(image)

# Normalize the image
  #normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
  #data[0] = normalized_image_array

# Predicts the model
        prediction = model1.predict(data)
        index = np.argmax(prediction)
        class_name = class_names1[index]
        confidence_score = prediction[0][index]

# Print prediction and confidence score
        print("hi")
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)

       
    # Now, you can use 'raw_image_data' as input to your neural network

    # Display the frame
        cv2.imshow('Webcam', frame)
    # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
# Release the webcam and close all OpenCV windows
    
import cv2
import time
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

data=np.ndarray(shape=(1,224,224,3),dtype=np.float32)
# Open a connection to the webcam (usually webcam index 0)
cap = cv2.VideoCapture(0)
# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
# Read and display frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Check if the frame is read successfully
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Preprocess the frame (resize, normalize, etc.) as needed for your neural network
    processed_frame = cv2.resize(frame, (224, 224)) # Example: Resize to 224x224
    # Convert the frame to a NumPy array (raw image data)
    raw_image_data = np.array(processed_frame)
    normalized_image_array = (raw_image_data.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

# Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    print(class_name[2:7])
    b=''.join(class_name[2:7])

    if confidence_score>0.51 and b == "Human":
  #np.set_printoptions(suppress=True)

# Load the model
        model1 = load_model("keras_model1.h5", compile=False)
        model1.summary()
# Load the labels
        class_names1 = open("labels1.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
  #data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
  #image = Image.open("IMG_1243.jpeg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
  #size = (224, 224)
  #image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
  #image_array = np.asarray(image)

# Normalize the image
  #normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
  #data[0] = normalized_image_array

# Predicts the model
        prediction = model1.predict(data)
        index = np.argmax(prediction)
        class_name = class_names1[index]
        confidence_score = prediction[0][index]

# Print prediction and confidence score
        print("hi")
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)

       
    # Now, you can use 'raw_image_data' as input to your neural network

    # Display the frame
        cv2.imshow('Webcam', frame)
    # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
# Release the webcam and close all OpenCV windows
    
import cv2
import time
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

data=np.ndarray(shape=(1,224,224,3),dtype=np.float32)
# Open a connection to the webcam (usually webcam index 0)
cap = cv2.VideoCapture(0)
# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
# Read and display frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Check if the frame is read successfully
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Preprocess the frame (resize, normalize, etc.) as needed for your neural network
    processed_frame = cv2.resize(frame, (224, 224)) # Example: Resize to 224x224
    # Convert the frame to a NumPy array (raw image data)
    raw_image_data = np.array(processed_frame)
    normalized_image_array = (raw_image_data.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

# Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    print(class_name[2:7])
    b=''.join(class_name[2:7])

    if confidence_score>0.51 and b == "Human":
  #np.set_printoptions(suppress=True)

# Load the model
        model1 = load_model("keras_model1.h5", compile=False)
        model1.summary()
# Load the labels
        class_names1 = open("labels1.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
  #data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
  #image = Image.open("IMG_1243.jpeg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
  #size = (224, 224)
  #image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
  #image_array = np.asarray(image)

# Normalize the image
  #normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
  #data[0] = normalized_image_array

# Predicts the model
        prediction = model1.predict(data)
        index = np.argmax(prediction)
        class_name = class_names1[index]
        confidence_score = prediction[0][index]

# Print prediction and confidence score
        print("hi")
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)

       
    # Now, you can use 'raw_image_data' as input to your neural network

    # Display the frame
        cv2.imshow('Webcam', frame)
    # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
# Release the webcam and close all OpenCV windows
    
import cv2
import time
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

data=np.ndarray(shape=(1,224,224,3),dtype=np.float32)
# Open a connection to the webcam (usually webcam index 0)
cap = cv2.VideoCapture(0)
# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
# Read and display frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Check if the frame is read successfully
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Preprocess the frame (resize, normalize, etc.) as needed for your neural network
    processed_frame = cv2.resize(frame, (224, 224)) # Example: Resize to 224x224
    # Convert the frame to a NumPy array (raw image data)
    raw_image_data = np.array(processed_frame)
    normalized_image_array = (raw_image_data.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

# Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    print(class_name[2:7])
    b=''.join(class_name[2:7])

    if confidence_score>0.51 and b == "Human":
  #np.set_printoptions(suppress=True)

# Load the model
        model1 = load_model("keras_model1.h5", compile=False)
        model1.summary()
# Load the labels
        class_names1 = open("labels1.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
  #data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
  #image = Image.open("IMG_1243.jpeg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
  #size = (224, 224)
  #image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
  #image_array = np.asarray(image)

# Normalize the image
  #normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
  #data[0] = normalized_image_array

# Predicts the model
        prediction = model1.predict(data)
        index = np.argmax(prediction)
        class_name = class_names1[index]
        confidence_score = prediction[0][index]

# Print prediction and confidence score
        print("hi")
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)

       
    # Now, you can use 'raw_image_data' as input to your neural network

    # Display the frame
        cv2.imshow('Webcam', frame)
    # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
# Release the webcam and close all OpenCV windows
    
import cv2
import time
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

data=np.ndarray(shape=(1,224,224,3),dtype=np.float32)
# Open a connection to the webcam (usually webcam index 0)
cap = cv2.VideoCapture(0)
# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
# Read and display frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Check if the frame is read successfully
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Preprocess the frame (resize, normalize, etc.) as needed for your neural network
    processed_frame = cv2.resize(frame, (224, 224)) # Example: Resize to 224x224
    # Convert the frame to a NumPy array (raw image data)
    raw_image_data = np.array(processed_frame)
    normalized_image_array = (raw_image_data.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

# Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    print(class_name[2:7])
    b=''.join(class_name[2:7])

    if confidence_score>0.51 and b == "Human":
  #np.set_printoptions(suppress=True)

# Load the model
        model1 = load_model("keras_model1.h5", compile=False)
        model1.summary()
# Load the labels
        class_names1 = open("labels1.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
  #data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
  #image = Image.open("IMG_1243.jpeg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
  #size = (224, 224)
  #image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
  #image_array = np.asarray(image)

# Normalize the image
  #normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
  #data[0] = normalized_image_array

# Predicts the model
        prediction = model1.predict(data)
        index = np.argmax(prediction)
        class_name = class_names1[index]
        confidence_score = prediction[0][index]

# Print prediction and confidence score
        print("hi")
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)

       
    # Now, you can use 'raw_image_data' as input to your neural network

    # Display the frame
        cv2.imshow('Webcam', frame)
    # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
# Release the webcam and close all OpenCV windows
    
import cv2
import time
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

data=np.ndarray(shape=(1,224,224,3),dtype=np.float32)
# Open a connection to the webcam (usually webcam index 0)
cap = cv2.VideoCapture(0)
# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
# Read and display frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Check if the frame is read successfully
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Preprocess the frame (resize, normalize, etc.) as needed for your neural network
    processed_frame = cv2.resize(frame, (224, 224)) # Example: Resize to 224x224
    # Convert the frame to a NumPy array (raw image data)
    raw_image_data = np.array(processed_frame)
    normalized_image_array = (raw_image_data.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

# Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    print(class_name[2:7])
    b=''.join(class_name[2:7])

    if confidence_score>0.51 and b == "Human":
  #np.set_printoptions(suppress=True)

# Load the model
        model1 = load_model("keras_model1.h5", compile=False)
        model1.summary()
# Load the labels
        class_names1 = open("labels1.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
  #data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
  #image = Image.open("IMG_1243.jpeg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
  #size = (224, 224)
  #image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
  #image_array = np.asarray(image)

# Normalize the image
  #normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
  #data[0] = normalized_image_array

# Predicts the model
        prediction = model1.predict(data)
        index = np.argmax(prediction)
        class_name = class_names1[index]
        confidence_score = prediction[0][index]

# Print prediction and confidence score
        print("hi")
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)

       
    # Now, you can use 'raw_image_data' as input to your neural network

    # Display the frame
        cv2.imshow('Webcam', frame)
    # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
# Release the webcam and close all OpenCV windows
    
