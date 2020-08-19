import cv2
import time
import argparse
import sys
import os
import numpy as np
from PIL import Image
from utils2 import *

from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.tensorflow_loader import load_tf_model, tf_inference
import thermal_monitor as tm

####################YOLOFACE CONFIG######################
parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
args = parser.parse_args()
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
#########################################################


####################MASKDETECT CONFIG####################
sess, graph = load_tf_model('models/face_mask_detection.pb')
# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
#########################################################



####################LOAD VIDEO###########################
vidcap = cv2.VideoCapture(0)
has_frame,frame = vidcap.read()
count = 0
while has_frame:
    for i in range (0, 10):  #FRAME SKIPPING
        has_frame,frame = vidcap.read()
    #print('Read a new frame: ', has_frame)
        
    #cv2.imshow("show", frame)
    #cv2.waitKey(1)


#### FACE GRAB
    
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                        [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))

    # Remove the bounding boxes with low confidence
    faces, imageList, pixelList = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    print('[i] ==> # detected faces: {}'.format(len(faces)))
    print('#' * 60)
    #if imageList != []:
    #    cv2.imwrite("test.jpg", imageList[0])



#### FACE TRACK

    visualizer = tm.visualizer.Visualizer()
    visualizer.run(frame)

####  MASK DETECT

    # Call inference() on each of the faces in imageList
    doneImage = []
    for i in range (0, len(imageList)):
        try:
            image = imageList[i]
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            conf_thresh = 0.5
            iou_thresh = 0.4
            target_shape = (260, 260)
            draw_result = True
            show_result = True

            output_info = []
            height, width, _ = image.shape
            image_resized = cv2.resize(image, target_shape)
            
            image_np = image_resized / 255.0  
            
            image_exp = np.expand_dims(image_np, axis=0)
            cv2.imshow('image',image_exp)
            y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)

            # remove the batch dimension, for batch is always 1 for inference.
            y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
            y_cls = y_cls_output[0]
            # To speed up, do single class NMS, not multiple classes NMS.
            bbox_max_scores = np.max(y_cls, axis=1)
            bbox_max_score_classes = np.argmax(y_cls, axis=1)

            # keep_idx is the alive bounding box after nms.
            keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                        bbox_max_scores,
                                                        conf_thresh=conf_thresh,
                                                        iou_thresh=iou_thresh,
                                                        )
        
            for idx in keep_idxs:
                conf = float(bbox_max_scores[idx])
                class_id = bbox_max_score_classes[idx]
                bbox = y_bboxes[idx]
                # clip the coordinate, avoid the value exceed the image boundary.
                xmin = max(0, int(bbox[0] * width))
                ymin = max(0, int(bbox[1] * height))
                xmax = min(int(bbox[2] * width), width)
                ymax = min(int(bbox[3] * height), height)

                if draw_result:
                    if class_id == 0:
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
                output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

            #cv2.imshow("show", image)
            #cv2.waitKey(1)
            #if show_result:
                #Image.fromarray(image).show()
                #Image2 = np.array(Image)
                #cv2.imwrite('one.jpg', Image2)  
                #cv2.imshow("show", Image2)
                #cv2.waitkey(1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            top  = pixelList[i][0]
            bot  = pixelList[i][1]
            left = pixelList[i][2]
            right= pixelList[i][3]
            frame[top:bot, left:right] = image
        except:
            print("Exception")
            continue
    cv2.imshow("show", frame)
    cv2.waitKey(1)

    '''
    # initialize the set of information we'll displaying on the frame
    info = [('number of faces detected', '{}'.format(len(faces)))]
    for (i, (txt, val)) in enumerate(info):
        text = '{}: {}'.format(txt, val)
        cv2.putText(frame, text, (10, (i * 20) + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
    '''



