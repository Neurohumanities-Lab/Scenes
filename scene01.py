import cv2
import mediapipe as mp
import numpy as np
import random
from pythonosc import udp_client

IP = "192.168.0.158"  # Escribe el IP aquí
PORT1 = 12000  # Escribe el primer puerto aquí
client = udp_client.SimpleUDPClient(IP, PORT1)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

Vis_threshold = 0.9     #Límite para visibilidad del landmark
Lateral_min = -1        #Límite mínimo para lateralidad
Lateral_max = 1         #Límite máximo para lateralidad
Proxim_min = -1         #Límite mínimo para proximidad
Proxim_max = 1          #Límite máximo para proximidad
Prof_lejos = 80         #Profundidad lejos (pixeles) - lejos
Prof_cerca = 230        #Proximidad cerca (pixeles) - cerca

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    #Get keypoints
    #image_rows, image_cols, _= image.shape
    height, width, _ = image.shape
    try:
      left_shoulder = results.pose_landmarks.landmark[11]
      right_shoulder = results.pose_landmarks.landmark[12]
      left_hip = results.pose_landmarks.landmark[23]
      right_hip = results.pose_landmarks.landmark[24]
      if ((left_shoulder.HasField('visibility') and left_shoulder.visibility > Vis_threshold) and
           (left_hip.HasField('visibility') and left_hip.visibility > Vis_threshold)):
          xls= int(left_shoulder.x*width) #left shoulder
          yls= int(left_shoulder.y*height)
          xlh= int(left_hip.x*width) #left hip
          ylh= int(left_hip.y*height)
          dS2Sl = np.sqrt((xls-xlh)**2 + (yls-ylh)**2) 
          left_vis = True
          print("left",dS2Sl)
      else:
         left_vis = False

      if ((right_shoulder.HasField('visibility') and right_shoulder.visibility > Vis_threshold) and
           (right_hip.HasField('visibility') and right_hip.visibility > Vis_threshold)):
          xrs= int(right_shoulder.x*width) #right shoulder
          yrs= int(right_shoulder.y*height)
          xrh= int(right_hip.x*width) #right hip
          yrh= int(right_hip.y*height)
          dS2Sr = np.sqrt((xrs-xrh)**2 + (yrs-yrh)**2) 
          right_vis = True
          print("right",dS2Sr)
      else:
         right_vis = False
      
      if right_vis and left_vis:
         medium_point = (xrs + xls)/2
         prof_med = (dS2Sl + dS2Sr)/2
         #print(medium_point)
         Laterality = ((Lateral_max-Lateral_min)/width) * (width - medium_point) + Lateral_min
         Profundity = ((Proxim_max-Proxim_min)/(Prof_cerca-Prof_lejos))*(prof_med-Prof_lejos) + Proxim_min
         #print("Ambos",Laterality)
         print("Ambos",Profundity)
         #print(width,height)
      else:
         if right_vis == True and left_vis == False:
            Laterality = ((Lateral_max-Lateral_min)/width) * (width - xrs) + Lateral_min
            Profundity = ((Proxim_max-Proxim_min)/(Prof_cerca-Prof_lejos))*(dS2Sr-Prof_lejos) + Proxim_min
            #print("Derecha", Laterality)
            print("Derecha", Profundity)
         if left_vis == True and right_vis == False:
            Laterality = ((Lateral_max-Lateral_min)/width) * (width - xls) + Lateral_min
            Profundity = ((Proxim_max-Proxim_min)/(Prof_cerca-Prof_lejos))*(dS2Sl-Prof_lejos) + Proxim_min
            #print("Izquierda", Laterality)
            print("Izquierda", Profundity)
            
     
      client.send_message("/Laterality", round(Laterality,2))
      client.send_message("/Proximity", round(Profundity,2))
    except:
      print("La persona no está completamente dentro del enfoque de la cámara.")

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()