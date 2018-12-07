# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:32:36 2018

@author: SPECTRE UNIT 01
"""

import cv2
import numpy as np
import cards
import time

cap = cv2.VideoCapture(1)
deck = cards.DeckData()

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

start_time = time.time()
frames = 1
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Gray, blur, and threshold the frame
    thresh = cards.preprocess_img(frame)    
    # Contour the binary image and identify rectangular contours as card
    cnts_sort, cnt_is_card = cards.find_cards(thresh)
    # Generate pairs of card images and their matches
    card_pairs = cards.process_cards(cnts_sort, cnt_is_card, frame, deck)
    
    
    
    # Display the resulting frame
    
    fps = round(frames/(time.time()-start_time), 1)
    frames += 1

    cv2.putText(frame,str(fps), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    
    cv2.imshow('frame',frame)
    cv2.imshow('thresh',thresh)
    if card_pairs is not None:
        # Generate a window composed of card pairs
        card_window = cards.generate_card_window(card_pairs)
        if card_window is not None:    
            cv2.imshow('card pairs', card_window)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
    