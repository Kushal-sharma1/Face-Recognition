import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("archive\haarcascade_frontalface_alt.xml")
collective=[]
dataset_path = " data\ "
name = input("ENTER THE NAME OF PERSON: \n")

skip = 0
while True :
    ret , frame = cap.read()
    if ret == False :
        continue

    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces, key = lambda f : f[2]*f[3])

    for x,y,w,h in faces[-1:] :
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        offset =10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section= cv2.resize(face_section ,(100,100))
        skip+=1
        if skip%10 == 0 :
            collective.append(face_section)
            print(len(collective))

    cv2.imshow("frame",frame)
    cv2.imshow("face section", face_section)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

#convert data int o numpy array
face_data =  np.asarray(collective)
print(face_data.shape)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)


#save data in form of numpy array
np.save(dataset_path.strip()+name+".npy",face_data)
print("data save sucessfully "+ dataset_path.strip()+name+".npy")

cap.release()
cv2.destroyAllWindows()
