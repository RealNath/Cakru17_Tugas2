import cv2
import numpy as np

video = cv2.VideoCapture("object_video.mp4")
bgSubstitution = cv2.createBackgroundSubtractorMOG2()
i=0
while video.isOpened():
    ret, frame = video.read()
    if ret: objMask = bgSubstitution.apply(frame)
    # cv2.imshow("Background Subtraction", objMask)
    # if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    contours, hierarchy = cv2.findContours(objMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frameContour = cv2.drawContours(frame, contours, -1, (0,255,0), 2)
    
    retval, tolerance = cv2.threshold(objMask, 180, 255, cv2.THRESH_BINARY)
    
    #Erosion, dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    erodedMask = cv2.morphologyEx(tolerance, cv2.MORPH_OPEN, kernel)

    #Adjust contour
    largeContours = [contour for contour in contours if cv2.contourArea(contour)>500]

    #Bounding box
    frameOut = frame.copy()
    for largeContour in largeContours:
        x,y,w,h = cv2.boundingRect(largeContour)
        frameOut = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,200), 3)

    cv2.namedWindow("final", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("final", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("final", frameOut)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

video.release()
cv2.destroyAllWindows()