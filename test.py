import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time

# Initialize the video capture object for the front camera
cap = cv2.VideoCapture(0)

# Variable to track if number plate has been detected
number_plate_detected = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Resize the frame to improve performance
    frame = imutils.resize(frame, width=500)
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Detect edges using Canny edge detection
    edged = cv2.Canny(gray, 170, 200)
    
    # Find contours in the edged image
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None
    
    # Iterate through contours to find the number plate
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break
    
    if NumberPlateCnt is not None:
        # Mask the part other than the number plate
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Configuration for tesseract
        config = ('-l eng --oem 1 --psm 3')
        
        # Run tesseract OCR on the image
        text = pytesseract.image_to_string(new_image, config=config)
        
        # Store the number plate and current time in CSV file only if number plate detected
        if text and not number_plate_detected:
            timestamp = time.asctime(time.localtime(time.time()))
            data = {'date': [timestamp], 'v_number': [text]}
            df = pd.DataFrame(data, columns=['date', 'v_number'])
            df.to_csv('data.csv', mode='a', header=False, index=False)
            
            # Set number_plate_detected to True to avoid duplicate entries
            number_plate_detected = True
            
            # Print recognized text and timestamp
            print("Number Plate:", text)
            print("Timestamp:", timestamp)
        
        # Show the frame with detected number plate
        cv2.imshow("Frame", frame)
        cv2.imshow("Number Plate", new_image)
    
    # Reset number_plate_detected to False if no number plate detected in current frame
    if NumberPlateCnt is None:
        number_plate_detected = False
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
