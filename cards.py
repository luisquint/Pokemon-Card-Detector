# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:35:46 2018

@author: SPECTRE UNIT 01
"""
import cv2
import numpy as np
from os.path import isfile, join, splitext 
from os import listdir

SCALER = 0.4
MAXWIDTH = round(700 * SCALER)
MAXHEIGHT = round(990 * SCALER)
CARD_MAX_AREA = 100000
CARD_MIN_AREA = 5000
BKG_THRESH = 80

class DeckData():
    
    MY_PATH = 'C:\\Users\SPECTRE UNIT 01\Desktop\School\Fall 2018\Digital Image Processing\Project\Template\\all_pokemon\\'
    DECKLIST_LOCATION = 'C:\\Users\SPECTRE UNIT 01\Desktop\School\Fall 2018\Digital Image Processing\Project\Template\metagross_solgaleo.txt'
        
    def __init__(self):
        
        self.card_images = self.load_card_images()
        self.card_keypoints = self.load_card_keypoints()        
        
    def load_card_images(self):
        card_img_names = get_cards_in_deck(self.DECKLIST_LOCATION)
        card_images = {}                
        for name in card_img_names:
            img = cv2.imread(self.MY_PATH+name+'.png')
            if img is None:
                img = cv2.imread(self.MY_PATH+name+'.jpg')
            card_images[name] = img
        
        return card_images
    
    def load_card_keypoints(self):
        sift = cv2.xfeatures2d.SIFT_create()
        card_keypoints = []
        
        i = 1        
        for name, img in self.card_images.items():
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(15,15),0)
            kp, des = sift.detectAndCompute(blur,None)
            card_keypoints.append([kp, des, name])
            print(i, name)
            i += 1
        
        return card_keypoints

def get_cards_in_deck(decklist_location):
    decklist = open(decklist_location, "r") 
    decklist_txt = decklist.read().splitlines()
    decklist_file_names = []
    for line in decklist_txt:
        if line != '':
            words = line.split(' ')
            if words[0].isdigit():
                words = words[1:]
                words = [element.lower() for element in words]
                if words[1] == 'energy':
                    decklist_file_names.append(gen_file_name(words[:-1]))
                else:
                    card_num = words.pop()
                    card_set = words.pop()
                    words.insert(0, card_set)
                    words.insert(0, card_num)
                    decklist_file_names.append(gen_file_name(words))
    return decklist_file_names

def gen_file_name(words):
    file_name = ''
    for word in words:
        file_name += word + '-'
    return file_name[:-1].replace("'", '')

def preprocess_img(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)    
    bkg_level = gray[100,100]
    thresh_level = bkg_level + BKG_THRESH
    retval, thresh_image = cv2.threshold(gray,thresh_level,255,cv2.THRESH_BINARY)
    
    return thresh_image

def find_cards(thresh):
    dummy,cnts,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)
    
    if len(cnts) == 0:
        return [], []
    
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)
    
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)

        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1
    return cnts_sort, cnt_is_card

def process_cards(cnts_sort, cnt_is_card, frame, deck):
    i = 0
    card_pairs = []
    for is_card in cnt_is_card:
        if is_card:
            contour = cnts_sort[i]
            # Find perimeter of card and use it to approximate corner points
            peri = cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,0.01*peri,True)
            pts = np.float32(approx)

            # Find width and height of card's bounding rectangle
            x,y,w,h = cv2.boundingRect(contour)

            # Warp card into SCALERx(900x770) flattened image using perspective transform
            img = flattener(frame, pts, w, h)
           
            # Run MATCH ALGORITHM HERE
            matched_card_name = match_card(img, deck)
            if matched_card_name is not None:
                matched_card_img = deck.card_images[matched_card_name]
                matched_card_img = cv2.resize(matched_card_img, (MAXWIDTH, MAXHEIGHT))
                numpy_horizontal_concat = np.concatenate((img, matched_card_img), axis=1)
            else:
                numpy_horizontal_concat = np.concatenate((img, img*0), axis=1)
            card_pairs.append(numpy_horizontal_concat)
        i += 1
        
    return card_pairs
        
def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down SCALERx(MAXWIDTHxMAXHEIGHT) 
    perspective. Returns the flattened, re-sized"""
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[MAXWIDTH-1,0],[MAXWIDTH-1,MAXHEIGHT-1],[0, MAXHEIGHT-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (MAXWIDTH, MAXHEIGHT))

    return warp

def match_card(card_img, deck):
    gray= cv2.cvtColor(card_img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray,None)
    
    bf = cv2.BFMatcher()
    max_good_points = 0
    max_pokemon_match = None

    for pokemon in deck.card_keypoints:
        des2 = pokemon[1]
        
        matches = bf.knnMatch(des1,des2, k=2)
    
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append([m])
        good_points = len(good)
        if good_points > max_good_points:
            max_good_points = good_points
            max_pokemon_match = pokemon[2]
            
    return max_pokemon_match
    
def generate_card_window(card_pairs):
    total_pairs = len(card_pairs)
    if total_pairs == 0:
        return
    if total_pairs == 1:
        return card_pairs[0]
    num_of_pairs = list(range(total_pairs))
    left_pairs = num_of_pairs[::2]
    right_pairs = num_of_pairs[1::2]
    if total_pairs % 2 == 1:
        right_pairs.append(total_pairs)
        card_pairs.append(card_pairs[0]*0)
    horizontal_imgs = []
    for i in range(len(left_pairs)):
        left_img = card_pairs[left_pairs[i]]
        right_img = card_pairs[right_pairs[i]]
        numpy_horizontal_concat = np.concatenate((left_img, right_img), axis=1)
        horizontal_imgs.append(numpy_horizontal_concat)
    window = horizontal_imgs[0]
    for i in range(1,len(horizontal_imgs)):
        img = horizontal_imgs[i]
        window = np.concatenate((window, img), axis=0)
    return window
