import cv2
import os
import code
import numpy as np
from PIL import Image
from ultralytics import YOLO
from plotter.Plotter import Plotter
from classify.Classifier import Classifier

sf = 0.3
track_model = YOLO('/home/raineai/Turtles/yolov5_turtles/20230430_yolov8x_turtlesonly_best.pt')
track_model.fuse()
classifier_weight = '/home/raineai/Turtles/yolov5_turtles/runs/train-cls/exp35/weights/best.pt' 
yolo_location = '/home/raineai/Turtles/yolov5_turtles'
classifier = Classifier(weights_file=classifier_weight, yolo_dir=yolo_location, confidence_threshold=0.3)
save_path = '/home/raineai/Turtles/datasets/classifyfort/train'
frame_no = '001430'
#frame_no = '000500'
image = '/home/raineai/Turtles/datasets/trim_vid/output/041219-0569AMsouth_trim/041219-0569AMsouth_trim_frame_'+frame_no+'.jpg'

img = cv2.imread(image)
imgw, imgh = img.shape[1], img.shape[0]
plotter = Plotter(imgw, imgh)
def GetTracks(frame, imgw, imgh):
        '''Given an image in a numpy array, find and track a turtle.
        Returns an array of numbers with class,x1,y1,x2,y2,conf,track id with x1,y1,x2,y2 all resized for the image'''
        box_array = []
        results = track_model.track(source = frame, stream=True, persist=True, boxes=True)
        for r in results:
                    boxes = r.boxes
                    for i, id in enumerate(boxes.id):
                        xyxyn = np.array(boxes.xyxyn[i,:])
                        box_array.append([int(boxes.cls[i]), int(float(xyxyn[0])*imgw), int(float(xyxyn[1])*imgh), 
                                        int(float(xyxyn[2])*imgw), int(float(xyxyn[3])*imgh), float(boxes.conf[i]), int(boxes.id[i])])
        return box_array
def Classify(frame, box_array, imgw, imgh):
        '''Given an image and box information around each turtle, clasify each turtle adding the conf and classification to the box array'''
        #code.interact(local=dict(globals(), **locals()))
        for box in box_array:
                xmin, xmax, ymin, ymax = max(0,box[1]), min(box[3],imgw), max(0,box[2]), min(box[4],imgh)
                cls_img_crop = frame[ymin:ymax,xmin:xmax]
                image = Image.fromarray(cls_img_crop)
                pred_list, predictions = classifier.classify_image(image) #classifiy it
                if box[6] ==36 or box[6]==57:#57:
                #    cv2.imshow('test', cls_img_crop)
                    print(box[6])
                    print(pred_list)
                    print(predictions)
                    #cv2.waitKey(0)
                p = int(pred_list[0])
                box.append((p+1)%2)
                box.append(predictions[p].item())
        #code.interact(local=dict(globals(), **locals()))
        return box_array
def CropnSave(box, img,classlabel,save_path,i):
    pad = 5
    x1,x2,y1,y2 = box[1],box[3],box[2],box[4]
    xmax,xmin,ymax,ymin = max(x1, x2),min(x1,x2),max(y1,y2),min(y1,y2)
    xminp,yminp = max(xmin-pad,1), max(ymin-pad,1)
    obj_img = img[yminp:ymax+pad,xminp:xmax+pad] 
    class_dir = os.path.join(save_path, classlabel)
    obj_filename =  '041219-0569AMsouth_trim_frame_'+frame_no+'_'+str(i)
    obj_path = os.path.join(class_dir, obj_filename+'.jpg')
    print(f'making file {obj_filename}.jpg in folder {class_dir}')
    cv2.imwrite(obj_path, obj_img)
    #code.interact(local=dict(globals(), **locals()))

box_array = GetTracks(img,imgw,imgh)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
box_array = Classify(img2,box_array,imgw,imgh)
#code.interact(local=dict(globals(), **locals()))

i = 1
if i==1:
    plotter.boxwithid(box_array,img)
    img = cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)
    cv2.imshow('images', img)
    #print(box_array)
    cv2.waitKey(0)
    
else:
    CropnSave(box_array[44],img,'turtle',save_path, 0)
    CropnSave(box_array[49],img,'turtle',save_path, 1)
    CropnSave(box_array[48],img,'turtle',save_path, 2)
    #CropnSave(box_array[46],img,'turtle',save_path, 3)
    #CropnSave(box_array[58],img,'turtle',save_path, 4)
    #CropnSave(box_array[87],img,'turtle',save_path, 5)

    CropnSave(box_array[33],img,'painted',save_path, 0)
    CropnSave(box_array[56],img,'painted',save_path, 1)
    #CropnSave(box_array[37],img,'painted',save_path, 2)
    #CropnSave(box_array[44],img,'painted',save_path, 3)
    #CropnSave(box_array[47],img,'painted',save_path, 4)
    #CropnSave(box_array[43],img,'painted',save_path, 5)


