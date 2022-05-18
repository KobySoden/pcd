#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

# python app.py 1 1 -1 video1.mp4 video2.mp4

#TODO need to shift the anchor frame if there are no matches to start
import os
from pickle import TRUE
import sys
import cv2
import time
import math
import signal
import youtube_dl
import numpy as np
import edit_distance
import tensorflow as tf
from imutils.video import FileVideoStream
from imutils.video import FPS
from utils import label_map_util
from utils import visualization_utils_color as vis_util

PIRATETHRESHOLD = .2
DEBUG_TIME = False
DEBUG_ALPHA = False
DEBUG_SKIPS = True
out = None
last_frame = None
last_time = None
out_fps = 30
video_num = 0
max_videos = 0
video_path_1 = 0
video_path_2 = 0
download_list = []
download_item = None
last_message = ''
frames_skipped = 0
recalculate_fps = False
cv2.ocl.setUseOpenCL(False)

#execution begins here
def compare_videos(path_video_1, path_video_2):
  global detection_graph, from_frame, recalculate_fps, out
  #object detection algorithm
  PATH_TO_CKPT = './ssd_inception2.pb'
  #file containing labeled objects
  PATH_TO_LABELS = './labels.pbtxt'
  thresh = 0.2                    #modifying this: default .2
  sequence_sorted = False
  store_output = True
  enable_tracking = True
  enable_detection = True
  adjust_frame = True
  adjust_perspective = True
  enable_tracking_template = True
  only_use_template_when_none = True
  enable_objects_threshold = False
  at_least_one_match = False
  recalculate_time = 0
  sequence_type = 'char'
  descriptor = "surf"
  tracker_type = 'KCF' # 'BOOSTING','MIL','KCF','TLD','MEDIANFLOW','GOTURN'
  NUM_CLASSES = 90
  MIN_MATCH_COUNT = 3             #modifying this: default 10
  SIMILARITY_THRESHOLD = 0.1    #modifying this: default .1
  trackers = {}
  positions = {}
  source_frame = 0
  ok = None
  font = cv2.FONT_HERSHEY_SIMPLEX
  size = 1
  weight = 2
  #default color encoding this varies for certain types of video
  color = (255,255,255)
  skips_max = 0
  skips_number = 0
  total_frames = 0
  
  last_message = ''

  #initialize tensorflow obj detection NN
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  detection_graph = tf.Graph()
  
  with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')
  
  with detection_graph.as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=detection_graph, config=config) as sess:
      total_time_init = time.time()
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      
      #feature extraction models available
      
      #sift = cv2.xfeatures2d.SIFT_create()
      surf = cv2.xfeatures2d.SURF_create()
      #fast = cv2.FastFeatureDetector_create()
      #orb = cv2.ORB_create()
      desc = surf
      show_points = 20

      #open videos with CV
      video_1 = cv2.VideoCapture(path_video_1)
      fps_1 = video_1.get(cv2.CAP_PROP_FPS)
      if DEBUG_TIME:
        print('fps_1', fps_1)
      video_2 = cv2.VideoCapture(path_video_2)
      fps_2 = video_2.get(cv2.CAP_PROP_FPS)
      if DEBUG_TIME:
        print('fps_2', fps_2)
      
      out = None
      
      #initialize states
      use_descriptor = True
      use_detection = False
      use_tracking = False
      matched_area = None
      frames_to_skip = 0
      processed_frames = 0

      #set start points for both videos being examined
      from_frame_1 = int(sys.argv[1])
      from_frame_2 = int(sys.argv[2])
      video_1.set(cv2.CAP_PROP_POS_FRAMES, from_frame_1)
      video_2.set(cv2.CAP_PROP_POS_FRAMES, from_frame_2)
      _, frame_1 = video_1.read()
      
      #declare objects and sequences of objects for both videos
      objects_1 = detect_objects(frame_1, thresh, detection_graph, sess, category_index, sequence_sorted=sequence_sorted, sequence_type=sequence_type)
      sequence_1 = objects_1['sequence']
      cv2.putText(frame_1, "skip: %s src: %s" % (processed_frames, sequence_1), (10, 30), font, size, color, weight)
      objects_2 = None
      area_2 = None
      sequence_2 = ''
      desc_kp_1, desc_des_1 = desc.detectAndCompute(frame_1, None)
      #print( desc.descriptorSize() )
      #print( desc_des_1.shape )
      
      #check if we should go to the end of the videos or if an endpoint is set
      until_end = False
      frame_num = int(sys.argv[3])
      if frame_num == -1:
        until_end = True
      
      #state machine of sorts
      while frame_num or until_end:
        total_frames += 1
        if recalculate_fps:
          if at_least_one_match:
            to_frame = from_frame_1 + math.ceil((time.time() - recalculate_time) * fps_1)
            if to_frame >= frame_num:
              break
        frame_num -= 1
        ok, frame_2 = video_2.read()
        if not ok:
          break

        if use_tracking:
          sequence_tmp = ''
          #iterate through all objects detected in video #2
          for object_2 in objects_2['objects']:
            #if we are tracking obj_2 update its tracker 
            #this is computer vision tracking
            if object_2['coords'] in trackers:
              start_time = time.time()
              ok, box = trackers[object_2['coords']].update(frame_2)
              box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
              elapsed_time = time.time() - start_time
              
              if DEBUG_TIME:
                print('tracking method', elapsed_time)  
              if ok:
                #add object to sequence of objects
                sequence_tmp += object_2['values'][sequence_type]
                #draw rectangle around the object
                cv2.rectangle(frame_2, (box[0], box[2]), (box[1], box[3]), (255, 0, 0), 2)
              
              #if we aren't tracking the object start tracking
              else:
                if enable_tracking_template:
                  process_static = True
                  if only_use_template_when_none:
                    #check object sequences in videos 1
                    num_matches = get_sequence_matches(sequence_1, sequence_tmp)
                    if num_matches > 0:
                      process_static = False
                  
                  if process_static:
                    start_time = time.time()
                    #search for object in frame
                    res = cv2.matchTemplate(frame_2, object_2['image'], cv2.TM_CCOEFF_NORMED)
                    elapsed_time = time.time() - start_time
                    if DEBUG_TIME:
                      print('tracking match', elapsed_time) 
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    threshold = 0.8
                    if max_val > threshold:
                      sequence_tmp += object_2['values'][sequence_type]
                      top_left = max_loc
                      h, w, _ = object_2['image'].shape
                      bottom_right = (top_left[0] + w, top_left[1] + h)
                      cv2.rectangle(frame_2, top_left, bottom_right, (0, 255, 0), 2)
          
          #write to cv display
          cv2.putText(frame_2, "alp: %s" % (sequence_2), (10, 30), font, size, color, weight)
          num_matches = get_sequence_matches(sequence_1, sequence_tmp)
          if num_matches > 0:
            if DEBUG_ALPHA:
              print_once('eq: %s ref: %s new: %s' % (num_matches, sequence_1, sequence_tmp))
          else:
            source_frame += processed_frames
            if DEBUG_SKIPS:
              print('skipped frames: %s' % (processed_frames))
            skips_number += 1
            skips_max = processed_frames if processed_frames > skips_max else skips_max
            if not recalculate_fps:
              video_1.set(cv2.CAP_PROP_POS_FRAMES, from_frame_1 + source_frame)
            else:
              to_frame = from_frame_1 + math.ceil((time.time() - recalculate_time) * fps_1)
              video_1.set(cv2.CAP_PROP_POS_FRAMES, to_frame)
            
            #read video frame in and detect objects in v1
            ok, frame_1 = video_1.read()
            objects_1 = detect_objects(frame_1, thresh, detection_graph, sess, category_index, sequence_sorted=sequence_sorted, sequence_type=sequence_type)
            sequence_1 = objects_1['sequence']
            cv2.putText(frame_1, "skip: %s src: %s" % (processed_frames, sequence_1), (10, 30), font, size, color, weight)
            
            desc_kp_1, desc_des_1 = desc.detectAndCompute(frame_1, None)
            #state transitions to find next match
            use_tracking = False
            use_detection = True
            use_descriptor = False
            processed_frames = 0

        if use_detection:
          if adjust_frame:
            area_2 = frame_2[matched_area[0]:matched_area[1],matched_area[2]:matched_area[3]]
          else:
            area_2 = frame_2

          #find objects in video 2
          objects_2 =  detect_objects(area_2, thresh, detection_graph, sess, category_index, matched_area=matched_area, sequence_sorted=sequence_sorted, sequence_type=sequence_type)
          sequence_2 = objects_2['sequence']
          cv2.putText(frame_2, "alp: %s" % (sequence_2), (10, 30), font, size, color, weight)
          #get the number of matching objects in the frames
          num_matches = get_sequence_matches(sequence_1, sequence_2)
          
          if DEBUG_ALPHA:
            print_once('eq: %s ref: %s new: %s' % (num_matches, sequence_1, sequence_2))
          
          if num_matches > 0:
            #reinitialize trackers
            trackers = {}
            were_coords_valid = False
            
            if enable_tracking:
              for object_2 in objects_2['objects']:
                if are_coords_valid(object_2['coords'], area_2.shape):
                    #create tracker for objects
                    trackers[object_2['coords']] = create_tracker(tracker_type)
                    if adjust_frame:
                      trackers[object_2['coords']].init(frame_2, object_2['global_coords'])
                    else:
                      trackers[object_2['coords']].init(frame_2, object_2['coords'])
                    were_coords_valid = True

            #lots of state transitions here 
            if were_coords_valid:
              if enable_tracking:
                use_tracking = True
                use_detection = False
                use_descriptor = False
              else:
                use_tracking = False
                use_detection = True
                use_descriptor = False
            else:
              use_tracking = False
              use_detection = True
              use_descriptor = False
          else:
            use_tracking = False
            use_detection = False
            use_descriptor = True
            
            if not enable_tracking:
              source_frame += processed_frames
              if DEBUG_SKIPS:
                print('detector skipped frames: %s' % (processed_frames))
              #skipping frame
              skips_number += 1
              skips_max = processed_frames if processed_frames > skips_max else skips_max
              
              if not recalculate_fps:
                video_1.set(cv2.CAP_PROP_POS_FRAMES, from_frame_1 + source_frame)
              else:
                to_frame = from_frame_1 + math.ceil((time.time() - recalculate_time) * fps_1)
                video_1.set(cv2.CAP_PROP_POS_FRAMES, to_frame)
              
              ok, frame_1 = video_1.read()
              cv2.putText(frame_1, "skip: %s" % (processed_frames), (10, 30), font, size, color, weight)
              desc_kp_1, desc_des_1 = desc.detectAndCompute(frame_1, None)
              processed_frames = 0        

        #descriptor aka feature extraction this is used to simplify processing
        if use_descriptor:
          matched_area = None
          descriptor_matched = False
          start_time = time.time()
          desc_kp_2, desc_des_2 = desc.detectAndCompute(frame_2, None)
          elapsed_time = time.time() - start_time
          
          if DEBUG_TIME:
            print(descriptor, elapsed_time)

          #use feature extraction and some sort of NN tree to determine matches
          if descriptor == "sift" or descriptor == "surf" or descriptor == "fast":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            start_time = time.time()
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            try:
              matches = flann.knnMatch(desc_des_1, desc_des_2, k=2)
            except:
              matches = []
            elapsed_time = time.time() - start_time
            if DEBUG_TIME:
              print('FLANN', elapsed_time)
            
            #count matches and add them to list if they are close enough to each other
            good = []
            for m,n in matches:
              if m.distance < 0.7*n.distance:
                good.append(m)
            area_2 = frame_2
            
            #calculate the similarity of the object positions
            similarity = 0
            if len(matches) > 0:
              similarity = len(good) / len(matches)
            
            #crazy vector math to transpose frames on each other if they are different
            #sizes, orientations or positions
            if len(good) > MIN_MATCH_COUNT:
              src_pts = np.float32([ desc_kp_1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
              dst_pts = np.float32([ desc_kp_2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
              M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
              matchesMask = mask.ravel().tolist()
              h,w,d = frame_1.shape
              pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
              try:
                dst = cv2.perspectiveTransform(pts,M)
                matched_area = get_rect_from_dst(dst, frame_2.shape)
                trans_coords = get_transformed_coords(dst, matched_area)
                frame_2 = cv2.polylines(frame_2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
                calc_height = matched_area[1] - matched_area[0]
                calc_width = matched_area[3] - matched_area[2]
                frame_height = frame_2.shape[0]
                frame_width = frame_2.shape[1]
                sim_rate = 1 + (((1 - (calc_height / frame_height)) + (1 - (calc_width / frame_width))) / 2)
                #increase similarity based on new transposed frames
                similarity *= sim_rate
                if similarity > SIMILARITY_THRESHOLD:
                  descriptor_matched = True
              except:
                pass
            else:
              if DEBUG_TIME:
                print( "Not enough matches were found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
              matchesMask = None

          if not descriptor_matched:
            if at_least_one_match:
              source_frame += processed_frames
              if DEBUG_SKIPS:
                print('descriptor skipped frames: %s' % (processed_frames))
              skips_number += 1
              skips_max = processed_frames if processed_frames > skips_max else skips_max
              if not recalculate_fps:
                video_1.set(cv2.CAP_PROP_POS_FRAMES, from_frame_1 + source_frame)
              else:
                to_frame = from_frame_1 + math.ceil((time.time() - recalculate_time) * fps_1)
                video_1.set(cv2.CAP_PROP_POS_FRAMES, to_frame)
              ok, frame_1 = video_1.read()
              cv2.putText(frame_1, "skip: %s" % (processed_frames), (10, 30), font, size, color, weight)
              desc_kp_1, desc_des_1 = desc.detectAndCompute(frame_1, None)
              processed_frames = 0
          else:
            if not at_least_one_match and recalculate_fps:
              recalculate_time = time.time()
            at_least_one_match = True
            if enable_detection:
              if adjust_frame:
                if is_matched_area_okay(trans_coords, frame_2.shape):
                  area_2 = frame_2[matched_area[0]:matched_area[1],matched_area[2]:matched_area[3]]
                  area_2 = cv2.polylines(area_2,[np.array(trans_coords)],True,255,3, cv2.LINE_AA)
                  if adjust_perspective:
                    area_2 = four_point_transform(area_2, trans_coords)
                else:
                  area_2 = frame_2  
              else:
                area_2 = frame_2
              objects_2 =  detect_objects(area_2, thresh, detection_graph, sess, category_index, matched_area=matched_area, sequence_sorted=sequence_sorted, sequence_type=sequence_type)
              sequence_2 = objects_2['sequence']
              cv2.putText(frame_2, "alp: %s" % (sequence_2), (10, 30), font, size, color, weight)
              num_matches = get_sequence_matches(sequence_1, sequence_2)
              if DEBUG_ALPHA:
                print_once('eq: %s ref: %s new: %s' % (num_matches, sequence_1, sequence_2))
              if num_matches > 0:
                use_descriptor = False
                use_detection = True
                use_tracking = False
              else:
                source_frame += processed_frames
                if DEBUG_SKIPS:
                  print('descriptor detector skipped frames: %s' % (processed_frames))
                skips_number += 1
                skips_max = processed_frames if processed_frames > skips_max else skips_max
                if not recalculate_fps:
                  video_1.set(cv2.CAP_PROP_POS_FRAMES, from_frame_1 + source_frame)
                else:
                  to_frame = from_frame_1 + math.ceil((time.time() - recalculate_time) * fps_1)
                  video_1.set(cv2.CAP_PROP_POS_FRAMES, to_frame)
                ok, frame_1 = video_1.read()
                desc_kp_1, desc_des_1 = desc.detectAndCompute(frame_1, None)
                objects_1 = detect_objects(frame_1, thresh, detection_graph, sess, category_index, sequence_sorted=sequence_sorted, sequence_type=sequence_type)
                sequence_1 = objects_1['sequence']
                processed_frames = 0
                use_descriptor = True
                use_detection = False
                use_tracking = False 

          #matches_img = cv2.drawMatches(frame_1, desc_kp_1, frame_2, desc_kp_2, good, None, **draw_params)

        if at_least_one_match:
          processed_frames += 1
        if matchesMask is None:
          matchesMask = []
        draw_params = dict(
             matchesMask = matchesMask[:show_points], # draw only inliers
             flags = 2)
        #print("%s of %s rate %s" % (len(good), len(matches), len(good)/len(matches)))
        try:
          matches_img = cv2.drawMatches(frame_1, desc_kp_1, frame_2, desc_kp_2, good[:show_points], None, **draw_params)
        except:
          matches_img = frame_1

        #store output to a file
        if store_output:
          if out == None:
            out = cv2.VideoWriter('out.avi', fourcc, 30.0, (matches_img.shape[1], matches_img.shape[0]), True)
          if not recalculate_fps:
            out.write(matches_img)
          else:
            video_insert(matches_img)
        
        cv2.imshow("Matches", matches_img)
        cv2.waitKey(1)
      
      if store_output:
        if out is not None:
          if recalculate_fps:
            video_close()
          out.release()
      
      #high level summary
      print('--- STATS ---')
      total_time = time.time() - total_time_init
      print('TOTAL TIME: ', total_time)
      print('TOTAL FRAMES: ', total_frames)
      print('SKIPS NUMBER: ', skips_number)
      print('MAX SKIP: ', skips_max)

      #exit and return value to a file
      f = open("piracy.txt", "w")
      if has_pirated_content(skips_number, total_frames):
        f.write("9000")
      else:
        f.write("0000")
      f.close()
      
#calculate number of matches
def percent_matched(num, denom):
    if denom <= 0:
      return 0
    return float(num)/float(denom)

def has_pirated_content(total_frames, skips_number, skips_max=1):
    #if our percent matches is greater than our threshold then we return true
    if percent_matched(skips_number, total_frames) > PIRATETHRESHOLD:
        #eventually we should use this skips max number for our calculation
        if skips_max > 0:
            return True
    return False 

def video_insert(frame):
  global out, last_frame, last_time
  if last_time is None:
    last_time = time.time()
  else:
    num_frames = math.floor((time.time() - last_time) * out_fps)
    last_time = time.time()
    for i in range(num_frames):
      out.write(last_frame)
  last_frame = frame

def video_close():
  global out, last_frame, last_time
  num_frames = math.floor((time.time() - last_time) * out_fps)
  for i in range(num_frames):
    out.write(last_frame)

def is_matched_area_okay(matched_area, frame_2_shape):
  return True

def print_once(message):
  global last_message
  if message != last_message:
    last_message = message
    print(last_message)

def youtube_download_hook(download):
  global download_item
  if download["status"] == "finished":
    print(download["filename"])
    video_num = download_item['index']
    os.rename(download["filename"], "internet%s.mp4" % (video_num))
    continue_downloads()

def load_from_youtube(video):
  ydl_opts = {"format": "mp4", "progress_hooks": [youtube_download_hook]}
  youtube_dl.YoutubeDL(ydl_opts).download([video])

def get_and_compare_videos(path_1, path_2, skip=True):
  global video_path_1, video_path_2, max_videos, download_list, recalculate_fps
  need_download = False
  video_path_1 = path_1

  #check for youtube downloads
  if 'http' in path_1:
    download_list.append({'source': path_1, 'index': 1})
    video_path_1 = 'internet1.mp4'
  elif path_1 == '0':
    video_path_1 = 0
    recalculate_fps = True
  video_path_2 = path_2

  #check for youtube download
  if 'http' in path_2:
    download_list.append({'source': path_2, 'index': 2})
    video_path_2 = 'internet2.mp4'
  elif path_2 == '0':
    video_path_2 = 0
    recalculate_fps = True
  if len(download_list) == 0:
    compare_videos(video_path_1, video_path_2)
  else:
    continue_downloads()

#function for downloading videos
def continue_downloads():
  global download_list, download_item, video_path_1, video_path_2
  if len(download_list) > 0:
    download_item = download_list.pop(0)
    load_from_youtube(download_item['source'])
  else:
    compare_videos(video_path_1, video_path_2)

#function to create object trackers
def create_tracker(tracker_type):
  if tracker_type == 'BOOSTING':
    return cv2.TrackerBoosting_create()
  elif tracker_type == 'MIL':
    return cv2.TrackerMIL_create()
  elif tracker_type == 'KCF':
    return cv2.TrackerKCF_create()
  elif tracker_type == 'TLD':
    return cv2.TrackerTLD_create()
  elif tracker_type == 'MEDIANFLOW':
    return cv2.TrackerMedianFlow_create()
  elif tracker_type == 'GOTURN':
    return cv2.TrackerGOTURN_create()
  else:
    return cv2.TrackerKCF_create()

#ensures coordinates of object are not too loarge
def are_coords_valid(box, orig):
  threshold = 0.8
  calc_height = ((box[1] - box[0])/orig[1])
  calc_width = ((box[3] - box[2])/orig[0])
  if calc_height >= threshold and calc_width >= threshold:
    return False
  return True

#detect objects in an image
def detect_objects(image, thresh, detection_graph, sess, category_index, matched_area=None, sequence_sorted=False, sequence_type='char'):
  image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image_np_expanded = np.expand_dims(image_np, axis=0)
  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
  boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  scores = detection_graph.get_tensor_by_name('detection_scores:0')
  classes = detection_graph.get_tensor_by_name('detection_classes:0')
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')
  start_time = time.time()
  
  if image_np_expanded[0] is not None:
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    elapsed_time = time.time() - start_time
    
    if DEBUG_TIME:
      print('cnn', elapsed_time)
    
    box = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        min_score_thresh=thresh,
        use_normalized_coordinates=True,
        line_thickness=4,
        sequence_sorted=sequence_sorted,
        sequence_type=sequence_type,
        matched_area=matched_area)
  else:
    box = {'sequence': '', 'objects': []}
  return box

def get_sequence_matches(sequence_1, sequence_2):
  if sequence_1 and sequence_2:
    sm = edit_distance.SequenceMatcher(a=sequence_1, b=sequence_2)
    sm.get_opcodes()
    sm.ratio()
    sm.get_matching_blocks()
    distance = sm.distance()
    num_matches = sm.matches()
    return num_matches
  else:
    return 0

def get_rect_from_dst(dst, orig):
  top = int(dst[0][0][1]) if dst[0][0][1] < dst[3][0][1] else int(dst[3][0][1])
  bottom = int(dst[1][0][1]) if dst[1][0][1] > dst[2][0][1] else int(dst[2][0][1])
  left = int(dst[0][0][0]) if dst[0][0][0] < dst[1][0][0] else int(dst[1][0][0])
  right = int(dst[2][0][0]) if dst[2][0][0] > dst[3][0][0] else int(dst[3][0][0])
  top = 0 if top < 0 else top
  left = 0 if left < 0 else left
  bottom = orig[0] if bottom > orig[0] else bottom
  right = orig[1] if right > orig[1] else right
  return (top, bottom, left, right)

def get_area_coords(dts):
  (top, bottom, left, right) = dts
  tl = (int(dts[0][0][0]), int(dts[0][0][1]))
  tr = (int(dts[3][0][0]), int(dts[3][0][1]))
  bl = (int(dts[1][0][0]), int(dts[1][0][1]))
  br = (int(dts[2][0][0]), int(dts[2][0][1]))
  return [tl, tr, br, bl]

def get_transformed_coords(dst, matched_area):
  (top, bottom, left, right) = matched_area
  tl = (-(left - int(dst[0][0][0])), -(top - int(dst[0][0][1])))
  tr = ((int(dst[3][0][0]) - left), -(top - int(dst[3][0][1])))
  bl = (-(left - int(dst[1][0][0])), int(dst[1][0][1]) - top)
  br = ((int(dst[2][0][0]) - left), (int(dst[2][0][1]) - top))
  return [tl, tr, br, bl]

def order_points(pts):
  # initialzie a list of coordinates that will be ordered
  # such that the first entry in the list is the top-left,
  # the second entry is the top-right, the third is the
  # bottom-right, and the fourth is the bottom-left
  rect = np.zeros((4, 2), dtype = "float32")
 
  # the top-left point will have the smallest sum, whereas
  # the bottom-right point will have the largest sum
  s = pts.sum(axis = 1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]
 
  # now, compute the difference between the points, the
  # top-right point will have the smallest difference,
  # whereas the bottom-left will have the largest difference
  diff = np.diff(pts, axis = 1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]
 
  # return the ordered coordinates
  return rect

def four_point_transform(image, pts):
  # obtain a consistent order of the points and unpack them
  # individually
  pts = np.array(pts)
  rect = order_points(pts)
  # rect = np.array(pts)
  (tl, tr, br, bl) = rect
 
  # compute the width of the new image, which will be the
  # maximum distance between bottom-right and bottom-left
  # x-coordiates or the top-right and top-left x-coordinates
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))
 
  # compute the height of the new image, which will be the
  # maximum distance between the top-right and bottom-right
  # y-coordinates or the top-left and bottom-left y-coordinates
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))
 
  # now that we have the dimensions of the new image, construct
  # the set of destination points to obtain a "birds eye view",
  # (i.e. top-down view) of the image, again specifying points
  # in the top-left, top-right, bottom-right, and bottom-left
  # order
  dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
 
  # compute the perspective transform matrix and then apply it
  M = cv2.getPerspectiveTransform(rect, dst)
  try:
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
  except:
    return image
  # return the warped image
  

def compare_2d_color_images(frame_1, frame_2):
  start_time = time.time()
  matches_num = 0
  height = frame_1.shape[0]
  width = frame_1.shape[1]
  size = width * height
  for i in range(height):
    for j in range(width):
      if frame_1[i][j][0] == frame_2[i][j][0] and frame_1[i][j][1] == frame_2[i][j][1] and frame_1[i][j][2] == frame_2[i][j][2]:
        matches_num += 1
  rate = matches_num / size
  elapsed_time = time.time() - start_time
  if DEBUG_TIME:
    print('iterate_2d', elapsed_time)

def compare_2d_gray_images(frame_1, frame_2):
  start_time = time.time()
  matches_num = 0
  gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
  gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
  height = gray_1.shape[0]
  width = gray_1.shape[1]
  size = width * height
  for i in range(height):
    for j in range(width):
      if gray_1[i][j] == gray_2[i][j]:
        matches_num += 1
  rate = matches_num / size
  elapsed_time = time.time() - start_time
  if DEBUG_TIME:
    print('iterate_2d', elapsed_time)

def compare_1d_gray_images(frame_1, frame_2):
  start_time = time.time()
  matches_num = 0
  gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
  flat_1 = [j for i in gray_1 for j in i]
  gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
  flat_2 = [j for i in gray_2 for j in i]
  size = len(flat_1)
  for i in range(size):
      if flat_1[i] == flat_2[i]:
        matches_num += 1
  rate = matches_num / size
  elapsed_time = time.time() - start_time
  if DEBUG_TIME:
    print('iterate_1d', elapsed_time)

#TODO figure out why youtube downloader is not working
#load_from_youtube()

get_and_compare_videos(sys.argv[4], sys.argv[5])

# Resolution: 1920 x 1080
#
# 2D color
# Operations = 1920 x 1080 x 3
# Time = 2.372
#
# 2D gray
# Operations = 1920 x 1080
# Time = 0.521
#
# SURF
# Time = 0.243
#
# CNN
# Time = 0.087
#
# Tracking
# Time = 0.003
#
# FLANN
# Size 1 = 4542
# Size 2 = 4117
# Time = 0.109
#
# BFMatcher
# Size 1 = 4542
# Size 2 = 4117
# Time = 0.164