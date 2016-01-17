#!/usr/bin/env python

'''
combine face detection with recognition
author:liuke
date:2015/12/07
'''

import sys

import cv2


def videoCap(src):
	cap=cv2.VideoCapture(src)		
	return cap

def frame_process(frame):
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	gray=cv2.equalizeHist(gray)
	return gray

def detect(frame, cascade):
	rects=cascade.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=4,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
	if len(rects)==0:
		return []
	rects[:,2:] += rects[:,:2]
	return rects

def draw_rects(frame,rects,color):
	cnt=0
	for x1,y1,x2,y2 in rects:
		cnt+=1
		cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
		#draw_str(frame,(x1,y1-5),'actor: x%d' %cnt )
		cv2.putText(frame, 'DR:x%d' %cnt , (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness = 2, lineType=cv2.LINE_AA)


def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


def main():
	cascade_src="/home/liuke/CV/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_alt.xml"
	

	cascade=cv2.CascadeClassifier(cascade_src)
	
	try:
		src=sys.argv[1]
	except:
		src=0
	cap=videoCap(src)

	while True:
		ret,frame=cap.read()
		gray=frame_process(frame)

		t=clock()
		rects=detect(gray,cascade)
		vis=frame.copy()
		draw_rects(vis,rects,(0,255,0))

		dt=clock()-t

		draw_str(vis,(20,20),'time: %.1f ms' % (dt*1000))
		cv2.imshow('facedetect',vis)

		if 0xFF & cv2.waitKey(5) == 27:
			break
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()