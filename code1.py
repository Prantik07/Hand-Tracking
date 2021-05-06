# Hand tracking using mediapipe library

import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)

# importing the hand detection module 
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # if any hand detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                
                h, w, _ = img.shape
                
                # getting the pixel value of the landmark
                cx, cy = int(lm.x*w), int(lm.y*h)

                if(id == 18):
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
    
    cv2.imshow('image', img)
    
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break
    


    
