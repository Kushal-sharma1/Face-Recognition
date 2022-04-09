import cv2
import numpy as np
import os

# KNN Algo for recognisation

def distance(xi,xf):
    d= np.sqrt(sum((xf-xi)**2))
    return d

def knn(data_train,query_point, k=5):
    x_t =data_train[:,:-1]
    y_t= data_train[:,-1]
    m = x_t.shape[0]
    value =[]
    for i in range(m):
        d1= distance(x_t[i], query_point)
        value.append([d1,y_t[i]])

    values = sorted(value , key=lambda x:x[0] )[:k]
    values = np.array(values)[:,-1]
    new_val = np.unique(values, return_counts =True)
    index = new_val[1].argmax()
    output= new_val[0][index]
    return output

# data preparation
skip=0
face_data= []
labels= []
dataset_path = " data\ "
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("archive\haarcascade_frontalface_alt.xml")
class_id = 0
names={}  #this is for maping between name and labels

for fx in os.listdir(dataset_path.strip()) :
    if fx.endswith('.npy'):
        data = np.load(dataset_path.strip()+fx)
        face_data.append(data)
        target = class_id*np.ones((data.shape[0],))
        labels.append(target)
        names[class_id] = fx[:-4]

        # names.insert(class_id ,fx[:-4])

        class_id+=1

face_dataset = np.concatenate(face_data ,axis =0)
labels_dataset = np.concatenate(labels ,axis= 0).reshape((-1,1))
print(face_dataset.shape)
print(labels_dataset.shape)
face_train = np.concatenate((face_dataset,labels_dataset),axis=1)
print(face_train.shape)


# testing

while True :
    ret ,frame = cap.read()
    if ret==False :
        continue

    faces = face_cascade.detectMultiScale(frame,1.3,5)

    for x,y,w,h in faces:
        offset =10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section= cv2.resize(face_section ,(100,100))

        out = knn(face_train , face_section.flatten())
        pred_name = names[int(out)]
        cv2.rectangle(frame ,(x,y),(x+w,y+h),(0,255.0),2)
        cv2.putText(frame , pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

    cv2.imshow("face_recgo", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
