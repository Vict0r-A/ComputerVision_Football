#This script shows both the pretrained model and the custom one we built

#import necessary dependencies
from ultralytics import YOLO
#already trained model to make predictions on new, unseen data.
INPUT_VIDEO_PATH = 'input_videos/test7.mp4'
model = YOLO("models/best.pt")
#print(model.model.names)returns a dictionary of classes
#.pt file is the extension for PyTorch model checkpoints
results = model(f"{INPUT_VIDEO_PATH}", save = True) #show = true shows frame by frame detections real time
#save saves it to a runs folder



# INPUT_VIDEO_PATH = 'input_videos/example_video.mp4'
# model = YOLO("best.pt")
# results = model(f"{INPUT_VIDEO_PATH}", save = True) 


#model.names = dictionary of the class number an the class value

#results produces a generator of results objects each result corresponds to a frame
#.boxes for bboxes, class ids, confidence
#.orig_img is just the original image (as a NumPy array) 