import pytesseract
import cv2
import pyttsx3



#FILE LOCATION AND COPYING
image = cv2.imread("Alphabet/word.png")
base_image = image.copy()
#######################


#CONVERTING IMAGE TO GRAYSCALE AND COMPRESSING IT
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3,50))
dilate = cv2.dilate(thresh, kernal, iterations=1)
################################################


#CONTOURING AND DILATING
cv2.imwrite("Alphabet/sample_dilate.png", dilate)
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])
###################################

#FOR LOOP FOR CONTOURS
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if h > 200 and w > 250:
        roi = base_image[y:]
        cv2.rectangle(image, (x,y), (x+w, y+h), (36, 255, 12), 2)
#########################################

#WRITING THE IMAGE
cv2.imwrite("Alphabet/sample_boxes.png", image)
###################

#OCR-ING THE IMAGE
ocr_result_original = pytesseract.image_to_string(base_image)
print(ocr_result_original)
#################


#TEXT TO SPEECH
engine = pyttsx3.init()
engine.say(ocr_result_original)
engine.runAndWait()
#############
