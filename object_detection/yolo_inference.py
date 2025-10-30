#This script shows both the pretrained model and the custom one we built

#import necessary dependencies
from ultralytics import YOLO
#already trained model to make predictions on new, unseen data.
INPUT_VIDEO_PATH = 'input_videos/test7.mp4'
model = YOLO("models/best.pt")
 
results = model(f"{INPUT_VIDEO_PATH}", save = True) 
#save saves it to a runs folder

 