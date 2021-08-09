from TEXT_DETECTION.load_textdetection import text_detection
from TEXT_RECOGNITION.load_textrecognition import text_recognition

import cv2
import numpy as np

text_detection = text_detection()
text_recognition = text_recognition()

img = cv2.imread('./sample.jpeg')
pred,_ = text_detection(img)

text_recog_result = text_recognition(img, pred)

for i in zip(pred, text_recog_result):
    x1,y1,x2,y2 = int(i[0][0][0]), int(i[0][0][1]), int(i[0][1][0]), int(i[0][2][1])
    color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
    img = cv2.rectangle(img, (x1,y1),(x2,y2),color,3)
    img = cv2.putText(img, i[1], (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,240), 2)
    cv2.imwrite("result.png", img)
