import cv2
import numpy as np
import PIL
from PIL import Image
import time
import glob
import os
import sys

input_keyboard = input("Please enter video or image: ")
object_classifier = cv2.CascadeClassifier("models/facial_recognition_model.xml") # an opencv classifier
basewidth = 300
index = 0

if input_keyboard == 'video':
	print("video")
	#videos = glob.glob(os.path.join('path'), recursive=True)
	#for video in videos:
		#index = index + 1
		#vid = cv2.imread(video)
		#gray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

		#faces = object_classifier.detectMultiScale(gray, 1.3, 5)
		#for (x,y,w,h) in faces:
			#cv2.rectangle(vid,(x,y),(x+w,y+h),(255,0,0),2)
			#roi_gray = gray[y:y+h, x:x+w]
			#roi_color = vid[y:y+h, x:x+w]
		#print(index)
		#cv2.imwrite('output/output-{0}.jpg'.format(index), vid)
		#print("Done")
		print("Video feature not yet ready")
		
elif input_keyboard == 'image':
	images = glob.glob(os.path.join('lfw','**','*.jpg'), recursive=True)

	for image in images:
		index = index + 1
		img = cv2.imread(image)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		faces = object_classifier.detectMultiScale(gray, 1.3, 5)
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]
		print(index)
		cv2.imwrite('output/output-{0}.jpg'.format(index), img)
		print("Done")
else:
    print("Error no format specified")

