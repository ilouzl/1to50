import json
import subprocess

import cv2
import numpy as np
import pyautogui
from keras.models import load_model


def get_model():
    return load_model('models/my_model_improved.h5')

def mouse_click(x, y):
    pyautogui.mouseDown(x, y, button='left')
    pyautogui.mouseUp(x, y, button='left')

def screen_capture():
    tmpFilename = "tmp.bmp"
    subprocess.call(["/usr/sbin/screencapture","-x", tmpFilename])
    im = cv2.imread(tmpFilename)
    return im

def predict_contour_digit(digit):
    im = np.zeros((40,40), dtype='uint8')
    o = 10
    try:
        im[o:o+digit.shape[0], o:o+digit.shape[1]] = digit
    except:
        print("Err")
        digit = digit[:,:-5]
        im[o:o+digit.shape[0], o:o+digit.shape[1]] = digit
        return 44
    im = cv2.resize(im, (28,28))
    # cv2.imwrite("digits/digit.bmp", im)
    p = model.predict(im.reshape(1,28,28,1))
    d = np.argmax(p)
    return int(d)

def search_block_values(blocks):
    for block in blocks:
        if block["value"] == next_number:
            found_block_procedure(block)
            return True
    return False

def found_block_procedure(block):
    print(block)
    block_cm = cm[block["cnt_index"]]
    mouse_click(block_cm[0] + roi[0], block_cm[1] + roi[1])
    mouse_click(block_cm[0] + roi[0], block_cm[1] + roi[1])

def load_grid_ccordinates():
    with open("grid_ccordinates.json", "r") as f:
        c = json.load(f)
    return [c["top_left_x"], c["top_left_y"], c["bottom_right_x"], c["bottom_right_y"]]


model = get_model()
blocks = []
next_number = 1
while next_number < 51:
    if search_block_values(blocks):
        next_number += 1
        continue

    img = screen_capture()
    print("capture")
    roi = load_grid_ccordinates()
    img = img[roi[1]:roi[3], roi[0]:roi[2]]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    cm =[]
    area = []
    bbox = []
    for c in contours:
        M = cv2.moments(c)
        cm.append((int(M['m10']/M['m00']), int(M['m01']/M['m00'])))
        area.append(cv2.contourArea(c))
        bbox.append(cv2.boundingRect(c))

    blocks = []
    digit = np.ones(len(contours))*-1
    for i, c in enumerate(contours):
        if 75 < area[i] < 1000:
            x,y,w,h = bbox[i]
            digit[i] = predict_contour_digit(thresh[y:y+h, x:x+w])
        elif 4000 < area[i] < 10000:
            block = {}
            block["cnt_index"] = i
            block["childs_idx"] = np.argwhere(hierarchy.reshape(-1,4)[:,3] == i).flatten()
            blocks.append(block)

    for block in blocks:
        if len(block["childs_idx"]) == 1:
            block["value"] = digit[block["childs_idx"][0]]
        elif len(block["childs_idx"]) == 2:
            # find leftmost child
            if bbox[block["childs_idx"][0]][0] < bbox[block["childs_idx"][1]][0]:
                block["value"] = digit[block["childs_idx"][0]]*10 + digit[block["childs_idx"][1]]
            else:
                block["value"] = digit[block["childs_idx"][0]] + digit[block["childs_idx"][1]]*10
    
    if search_block_values(blocks):
        next_number += 1
