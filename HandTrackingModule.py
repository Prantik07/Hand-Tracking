''' Hand Tracking using mediapipe library '''

import cv2
import mediapipe as mp
import time


class handDetector():
    
    def __init__(self, mode=False, maxHands=2, detection_confidence = 0.5, tracking_confidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detection_confidence=detection_confidence
        self.tracking_confidence=tracking_confidence
                
        # importing the hand detection module 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 
                                        self.detection_confidence, self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

        
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        # if any hand detected
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                
        return img
    
    
    def findPosition(self, img, handNo=0, draw=True):
        lmlist = [] # list of all hand landmarks
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, _ = img.shape
                # getting the pixel value of the landmark
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
                    
        return lmlist

                
                

def main():
    
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        _, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if(len(lmlist)):
            print(lmlist[4])
        
        cv2.imshow('image', img)
    
        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q'):
            break
        


if __name__ == '__main__':
    main()
    
