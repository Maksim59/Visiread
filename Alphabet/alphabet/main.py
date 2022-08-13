import easyocr
import pyttsx3


#LANGUAGES
reader = easyocr.Reader(['en'])
########


#DIRECTORY
results = reader.readtext('0.png')
##########

#ITERATION
text = ''
for result in results:
    text += result[1] + ' '
##################

#SPEAKING ASPECT
engine = pyttsx3.init()
engine.say(text)
engine.runAndWait()
##################
