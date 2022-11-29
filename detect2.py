from importlib.resources import path
from unittest import result
import cv2
import torch
import numpy



def position( frame,results):
    for rsl  in results:
        #confidence = rsl['name']
        #clas = rsl ['class']
        x1 = int(rsl['xmin'])
        y1 = int(rsl['ymin'])
        x2 = int(rsl['xmax'])
        y2 = int(rsl['ymax']) 
        print(x1, y1, x2, y2)
        cropped_image = frame[y1:y2, x1:x2]

        return cropped_image

def plate(frame, model, model2):

    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    # frame = cv2.imread('image3.jpg')

    # model2 = torch.hub.load('ultralytics/yolov5', 'custom', path='bestfinal.pt')

    detections = model(frame)

    results = detections.pandas().xyxy[0].to_dict(orient = "records")
    x = numpy.array(results)
    #print(x)

    cropped_image = position(frame,results)
    img = position(frame,results)
    m,n,_ = cropped_image.shape
    #cv2.imshow('img',cropped_image)
    detections = model2(cropped_image)
    results = detections.pandas().xyxy[0].to_dict(orient = "records")
    s = numpy.array(results)
    #print(x)

    #cropped_image = cv2.resize(cropped_image, (540, 540),interpolation = cv2.INTER_NEAREST)
    for rsl  in results:
            confidence = rsl['name']
            #clas = rsl ['class']
            x1 = int(rsl['xmin'])
            y1 = int(rsl['ymin'])
            x2 = int(rsl['xmax'])
            y2 = int(rsl['ymax']) 

            cv2.rectangle(cropped_image,(x1,y1),(x2,y2),(255,0,0),2)

            cv2.putText(cropped_image,str(confidence),(x1+10,y1+10),cv2.FONT_HERSHEY_DUPLEX,1,(60,255,255),1)
    print('ket qua truoc khi sap xep', s)

    a = sorted(s,key = lambda x: x['xmin'])
    plate = ""
    print("ti le ", m/n)
    print("hieu", (a[3]['xmin'] - a[2]['xmin']) )
    if (m/n < 0.45):
        
        for i in a:
            plate+=i['name']
    elif( (a[3]['xmin'] - a[2]['xmin']) < 18 ):
        a = sorted(s,key = lambda x: x['ymin'] )
        b = sorted([a[0],a[1],a[2],a[3]],key = lambda x: x['xmin'] )
        for i in range(4):
            plate += b[i]['name']
        c = sorted([a[4],a[5],a[6],a[7],a[8]],key = lambda x: x['xmin'] )
        for i in range(5):
            plate += c[i]['name']

    else :
        a = sorted(s,key = lambda x: x['ymin'] )
        b = sorted([a[0],a[1],a[2]],key = lambda x: x['xmin'] )
        for i in range(3):
            plate+= b[i]['name']
        c = sorted([a[3],a[4],a[5],a[6],a[7]],key = lambda x: x['xmin'] )
        for i in range(5):
            plate+= c[i]['name']

    print("bien so", plate)


    cropped_image = cv2.resize(cropped_image, (200, 200),interpolation = cv2.INTER_NEAREST)



    #cv2.imshow('img2',cropped_image)


    #cv2.waitKey(0)

    return plate,img,cropped_image

