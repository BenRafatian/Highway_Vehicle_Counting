import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
import core.utils as utils
from core.config import cfg
# from core.yolov4 import filter_boxes
import numpy as np
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto



# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
# Flag metrics
input_size = 416
video_path = './data/video/Highway.mp4'
weights = './checkpoints/yolov4-tiny-416'
model = 'yolov4'
iou = 0.45 # iou threshold
score = 0.50 # score threshold


cap = cv2.VideoCapture(video_path)

# Input video metrics
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# prepares the output video
out = cv2.VideoWriter('./output/output.avi',cv2.VideoWriter_fourcc('M','J','P','G'),fps , (width, height))

def is_in_boundary(cords, bound):
    if cords[0] < bound[0]:
        return False
    elif cords[1] < bound[1]:
        return False
    elif cords[2] > bound [2]:
        return False
    elif cords[3] > bound[3]:
        return False
    else:
        return True                

def car_counter(track, bound):
    if not is_in_boundary(track.to_tlbr(), bound):
        return 0
    else:
        return track.track_id


def center_counter(xmin, ymin, xmax, ymax):
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    return int(cx), int(cy)
# Definition of the parameteres
max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0

###############################
total_cars = 0

carscrossed1 = set()
carscrossed2 = set()
carscrossed3 = set()
carscrossedopposite = set()



# boundarys for seperated counting
lane_1 = [345, 230, width, height]
lane_2 = [220, 230, 345, height]
lane_3 = [0, 230, 220, height]
opposite = [450, 90, width, 200]

lane3_x = 220
lane2_x = lane3_x + 125
lane1_x = lane2_x + 180
###############################

# initialize deep sort
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric)




# load standard tensorflow saved model
saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

# load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True
STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
anchors = np.array(cfg.YOLO.ANCHORS_TINY)
ANCHORS = anchors.reshape(2, 3, 2)
XYSCALE = cfg.YOLO.XYSCALE_TINY if model == 'yolov4' else [1, 1]

total_id = 0

frame_number = 0
start_time = time.time()
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_number +=1
    # print('Frame #: ', frame_number)

    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    loop_start_time = time.time()

    # run detection
    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score
    )

    # convert data to numpy arrays and slice out unused elements
    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0]
    bboxes = bboxes[0:int(num_objects)]
    scores = scores.numpy()[0]
    scores = scores[0:int(num_objects)]
    classes = classes.numpy()[0]
    classes = classes[0:int(num_objects)]

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
    original_h, original_w, _ = frame.shape
    bboxes = utils.format_boxes(bboxes, original_h, original_w)

    # store all predictions in one parameter for simplicity when calling functions
    pred_bbox = [bboxes, scores, classes, num_objects]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    
    # we want the car class only
    allowed_classes = ['car']

    names = []
    deleted_indx = []
    for i in range(num_objects):
        class_indx = int(classes[i])
        class_name = class_names[class_indx]
        if class_name not in allowed_classes:
            deleted_indx.append(i)
        else:
            names.append(class_name)
    names = np.array(names)
    count = len(names)    
    
    
    # delete detections that are not in allowed_classe
    bboxes = np.delete(bboxes, deleted_indx, axis=0)
    scores = np.delete(scores, deleted_indx, axis=0)

    # encode yolo detections and feed to tracker
    features = encoder(frame, bboxes)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

    #initialize color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # run non-maxima supression
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]       

    # Call the tracker
    tracker.predict()
    tracker.update(detections)

    # now that we have tracking system is time to count





    # update tracks
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue 
        bbox = track.to_tlbr()
        class_name = track.get_class()
        
        carscrossed1.add(car_counter(track, lane_1)) 
        carscrossed2.add(car_counter(track, lane_2)) 
        carscrossed3.add(car_counter(track, lane_3))                  
        carscrossedopposite.add(car_counter(track, opposite))
        if track.track_id > total_id:
            total_id = track.track_id
    # draw bbox on screen
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.circle(frame, center_counter(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), 3, (255,0,0), -1)
        # for writing the tracked item id while displaying
        # cv2.putText(frame, str(track.track_id),center_counter(bbox[0], bbox[1], bbox[2], bbox[3]),0, 0.5, (0,255,0),2)
        # cv2.circle(frame, center_counter(bbox[0], bbox[1], bbox[2], bbox[3]), 3, (255,0,0), -1)
    # if enable info flag then print details about each track

    # print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

    #draw boundaries
    cv2.rectangle(frame,(lane_1[0], lane_1[1]), (lane_1[2], lane_1[3]), (100,255,100), 1)
    cv2.rectangle(frame,(lane_2[0], lane_2[1]), (lane_2[2], lane_2[3]), (100,255,100), 1)
    cv2.rectangle(frame,(lane_3[0], lane_3[1]), (lane_3[2], lane_3[3]), (100,255,100), 1)
    cv2.rectangle(frame,(opposite[0], opposite[1]), (opposite[2], opposite[3]), (100,255,100), 1)

    total_cars = len(carscrossed1) + len(carscrossed2) + len(carscrossed3) + len(carscrossedopposite) - 4
    # calculate frames per second of running detections
    temp_fps = int(1.0 / (time.time() - loop_start_time))
    cv2.putText(frame, "FPS: " + str(temp_fps), (0, height-5), 0, 0.6, (200, 200, 0), 1)
    cv2.putText(frame,"Lane1: "+ str(len(carscrossed1) - 1) , (0, height-25), 0, 0.6, (200, 200, 0), 1)
    cv2.putText(frame,"Lane2: "+ str(len(carscrossed2) - 1) , (0, height-45), 0, 0.6, (200, 200, 0), 1)
    cv2.putText(frame,"Lane3: "+ str(len(carscrossed3) - 1) , (0, height-65), 0, 0.6, (200, 200, 0), 1)
    cv2.putText(frame,"opposite: "+ str(len(carscrossedopposite) - 1) , (0, height-85), 0, 0.6, (200, 200, 0), 1)
    cv2.putText(frame,"Total Cars : "+ str(total_cars) , (0, height-105), 0, 0.6, (200, 200, 0), 1)
    



    result = np.asarray(frame)
    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Output Video", result)
    out.write(result)

    if cv2.waitKey(1) == 27:
        break
cap.release()
out.release()

average_fps =  int(frame_number / (time.time() - start_time) ) 

print("Vehicles crossed Lane 1: " + str(len(carscrossed1) - 1))
print("Vehicles crossed Lane 2: " + str(len(carscrossed2) - 1))
print("Vehicles crossed Lane 3: " + str(len(carscrossed3) - 1))
print("Vehicles crossed opposite side: " + str(len(carscrossedopposite) - 1))
print("Total cars: " + str(total_cars))
print("Average FPS: " + str(average_fps))

# this is the actual total number of counted cars
print("total ID: " + str(total_id))
cv2.destroyAllWindows()