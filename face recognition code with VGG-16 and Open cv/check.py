import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


#--------------------------------------------------------------------------------
face_detect = cv2.CascadeClassifier("E:/face recognition code with VGG-16 and Open cv/face recognition code with VGG-16 and Open cv/haarcascade_frontalface_default.xml")

def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3,5)
    if len(faces)==0:
        return None
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h , x: x+w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return cropped_face

#--------------------------------------------------------------------------------

# Load model from saved file
model = load_model('face_recog.h5')

# Load class names
class_names = ['an','lam', 'thuan']  # Thay thế bằng tên lớp của bạn

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from camera
    ret, frame = cap.read()
    
    # Extract face from frame
    face = face_extractor(frame)
    if face is not None:
        face = cv2.resize(face, (100, 100))

        # Resize face to match input size of model
        img = cv2.resize(face, (244, 244))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)

        # Make prediction
        preds = model.predict(x)
        class_idx = np.argmax(preds)
        class_name = class_names[class_idx]

        # Draw class name and face on frame
        cv2.putText(frame, class_names[class_idx], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



    # Show frame
    cv2.imshow('frame', frame)
    
    # Press 'q' to exitq
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Release camera and close window
cap.release()
cv2.destroyAllWindows()