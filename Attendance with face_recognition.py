import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


print('''Attendance with Face Recognition\n''')

path = 'images'
images = []
personNames = []
roll_dictionary = {"Name":"Roll_Number","Unknown":" "}
myList = os.listdir(path)
# print(myList)
for current_image in myList:
    current_encoded_image = cv2.imread(f'{path}/{current_image}')
    images.append(current_encoded_image)
    personNames.append(os.path.splitext(current_image)[0])
# print(personNames)
print("Encoding start...")

def faceEncodings(images):
    encodeList = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodeList.append(encode)
    return encodeList


def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if person_name in nameList:
            f.writelines(f'')
        if person_name not in nameList:
            time_now = datetime.now()
            Time_str = time_now.strftime('%I:%M:%p')
            Date_str = time_now.strftime('%d/%b/%Y')
            Day_str = time_now.strftime('%A')
            f.writelines(f'\n{name},{roll_number},{Time_str},{Day_str},{Date_str}')


encodeListKnown = faceEncodings(images)
print("All Encodings Done!")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # ret is boolean variable return true if frame is available
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLocation in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            person_name = personNames[matchIndex]
            roll_number = roll_dictionary[person_name]
            name = person_name
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)
        else:
            person_name = "Unknown"
            roll_number = " "
            name = person_name
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 19, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (255, 19, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)


    cv2.imshow('Attendance With Face Recognition By TAHIR HABIB', frame)
    if cv2.waitKey(10) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
