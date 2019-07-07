import cv2
from ctypes import *
import math
import random
import os
import time
from playsound import playsound
import numpy as np


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/akhilpatil/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def capture():
    camera = cv2.VideoCapture(0)
    for i in range(10):
        return_value, image = camera.read()
        cv2.imwrite('opencv'+'.png', image)
    del(camera)

def send(str1):
    print("in send")
    cmd ="curl -H \"Content-Type: application/text\" -X POST -d " + str1 + " http://10.10.13.158:8080/data/"
    os.system(cmd)
  
if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]

    capture()
    net = load_net("/home/akhilpatil/darknet/cfg/yolov3.cfg", "/home/akhilpatil/darknet/yolov3.weights", 0)
    meta = load_meta("cfg/coco.data")
    r = detect(net, meta, "opencv.png")
    str1 = ['0','0','0','0']
    f = open('light.txt', 'r')
    str2 = f.read()
    for i in range(0,4):
        str1[i]=str2[i]    
    print(str1)
    print(type(str1))
    f.close()
    f1=False 
    f2=False
    f3=False
    f4=False
    c1=0
    c2=0
    c3=0
    c4=0
    print (type(r[0]))
    for i in range(0,len(r)):
        if r[i][0] == 'person':
            x=r[i][2][0]
            y=r[i][2][1]
            print ('x', r[i][2][0])
            print ('y', r[i][2][1])  
            print ("....................................................")
            if(x<=240 and y>=170):
                str1[0]='1'
                f1=True
                print ("part 1")
                c1=c1+1
                continue
            '''elif f1 == True:
                if c1>0:
                    c1=c1-1
                str1[0]='0'
                f1 =False
                continue'''
            if(x>270 and x<=415 and y>=155 and y<=400):
                str1[1]='1'
                f2=True               
                print ("part 2")
                c2=c2+1
                continue
            '''elif f2 == True:
                if c2>0:
                    c2=c2-1
                str1[1]='0'
                f2 =False
                continue'''
            if(x>415 and x<=510 and y>=90 and y<=340):
                str1[2]='1'
                f3=True
                print ("part 3")
                c3=c3+1
                continue
            '''elif f3 == True:
                if c3>0:
                    c3=c3-1
                str1[2]='0'
                f3 =False
                continue'''
            if(x>510 and y>=95 and y<330):
                str1[3]='1'
                f4=True
                print ("part 4")
                c4=c4+1
                continue
            '''elif f4 == True:
                if c4>0:
                    c4=c4-1
                str1[3]='0'
                f4 =False
                continue '''
    if f1== False:
        str1[0]='0'
    if f2== False:
        str1[1]='0'
    if f3== False:
        str1[2]='0'
    if f4== False:
        str1[3]='0'
    str1="".join(str1)
    print(type(str1))
    print("str",str1)
    if c1>=4 or c2>=4 or c3>=4 or c4>=4:
        print ("chaos",c1,c2,c3,c4)
        playsound('keepquite.mp3')
        img = cv2.imread('aa.jpg')
        res_img=cv2.resize(img, (1280,685))
        cv2.imshow('sample image',res_img)
        cv2.waitKey(0) # waits until a key is presse
        time.sleep(5)
        cv2.destroyAllWindows() # destroys the window showing image

    send(str1) 
    #time.sleep(20)
    print("sending....")
    f= open("light.txt","w")
    f.write(str1)
    f.close() 
    #print r
    print ("Sleeping....")
    #time.sleep(20)
    #str1="".join(str1)
    #




