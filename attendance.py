import face_recognition as fr
import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
from datetime import datetime
import numpy as np



def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    # print(encodeList)
    return encodeList


def encode_faces(folder):
    list_people_encoding=[]

    for filename in os.listdir(folder):
        known_image=fr.load_image_file(f'{folder}{filename}')
        known_encoding=fr.face_encodings(known_image)

        if len(known_encoding) > 0:
            list_people_encoding.append((known_encoding[0],filename))
        else:
            print(f'No face found in {filename}')
    return list_people_encoding



# def encode_faces(folder):
#     list_people_encoding=[]
#
#     for filename in os.listdir(folder):
#         known_image=fr.load_image_file(f'{folder}{filename}')
#         known_encoding=fr.face_encodings(known_image)[0]
#         list_people_encoding.append((known_encoding,filename))
#     return list_people_encoding
def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')


def find_target_face():
    face_location=fr.face_locations(target_image)
    for person in encode_faces('images/'):
        encode_face=person[0]
        filename=person[1]

        is_target_face=fr.compare_faces(encode_face,target_encoding,tolerance=0.6)
        print(f'{is_target_face} {filename}')

        if face_location:
            face_number=0
            for location in face_location:
                if is_target_face[face_number]:
                    label=filename[:len(filename)-4]
                    attendance(label)
                    create_frame(location,label)
                face_number+=1

def create_frame(location,label):
    top,right,bottom,left=location
    cv.rectangle(target_image,(left,top),(right,bottom),(255,0,0),2)
    cv.rectangle(target_image,(left,bottom+20),(right,bottom),(255,0,0),cv.FILLED)
    cv.putText(target_image,label,(left+3,bottom+14),cv.FONT_HERSHEY_DUPLEX, 0.4 ,(255,255,255),1)

def render_image():
    rgb_img=cv.cvtColor(target_image,cv.COLOR_BGR2RGB)
    cv.imshow('Face Recognition',rgb_img)
    cv.waitKey(0)

choice=2
if(choice==1):
   Tk().withdraw()
   load_image=askopenfilename()
   target_image=fr.load_image_file(load_image)
   target_encoding=fr.face_encodings(target_image)
   find_target_face()
   render_image()
else:
    path = 'images'
    images = []
    personNames = []
    myList = os.listdir(path)
    print(myList)
    for cu_img in myList:
        current_Img = cv.imread(f'{path}/{cu_img}')
        images.append(current_Img)
        personNames.append(os.path.splitext(cu_img)[0])
    print(personNames)
    encodeListKnown = faceEncodings(images)
    print('All Encodings Complete!!!')
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        faces = cv.resize(frame, (0, 0), None, 0.25, 0.25)
        faces = cv.cvtColor(faces, cv.COLOR_BGR2RGB)

        facesCurrentFrame = fr.face_locations(faces)
        encodesCurrentFrame = fr.face_encodings(faces, facesCurrentFrame)

        for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
            matches = fr.compare_faces(encodeListKnown, encodeFace)
            faceDis = fr.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = personNames[matchIndex].upper()
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv.FILLED)
                cv.putText(frame, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                attendance(name)

        cv.imshow('Webcam', frame)
        if cv.waitKey(1) == 13:
            break

    cap.release()
    cv.destroyAllWindows()