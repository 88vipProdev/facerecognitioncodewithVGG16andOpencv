import os 
import cv2

face_detect = cv2.CascadeClassifier("E:/face recognition code with VGG-16 and Open cv/face recognition code with VGG-16 and Open cv/haarcascade_frontalface_default.xml")

def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3,5)
    if len(faces)==0:
        return None
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h , x: x+w]
    return cropped_face

person_name = input("Enter person's name: ")
no_images = 0 

if not os.path.exists("E:/face recognition code with VGG-16 and Open cv/face recognition code with VGG-16 and Open cv/image/{}".format(person_name)):
    os.makedirs("E:/face recognition code with VGG-16 and Open cv/face recognition code with VGG-16 and Open cv/image/{}".format(person_name))

video = cv2.VideoCapture(0)

while True:
    ret,frame=video.read()
    if face_extractor(frame) is not None:
        no_images+=1 
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        cv2.imwrite("E:/face recognition code with VGG-16 and Open cv/face recognition code with VGG-16 and Open cv/image/{}/{}_{}.jpg".format(person_name, person_name, no_images),face)
        cv2.putText(face,str(no_images),(40,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("Frame",face)
    else:
        print("Something Went Wrong")
    if  cv2.waitKey(1)==27:
        break
    if no_images >= 200: # kiểm tra số lượng ảnh đã chụp và dừng lại nếu đã đủ
        break

video.release()
cv2.destroyAllWindows()
