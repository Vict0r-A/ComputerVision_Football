from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import supervision as sv
import cv2
import numpy as np

#{0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
class ObjectDetection:
    def __init__(self, model_path ="models/best.pt"):
        self.model = YOLO(model_path)
    
    
    def detect(self, frame):
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections
    
   
    
    def filter_players(self, detections):
        # Reassign goalkeeper â†’ player
        class_remap = {1: 2}
        detections.class_id = np.array(
            [class_remap.get(cls, cls) for cls in detections.class_id],
            dtype=int
        )
        # Keep only players
        mask = detections.class_id == 2
        return detections[mask]

    def filter_ball(self, detections):
        mask = detections.class_id == 0  # keep only ball
        return detections[mask]

    def filter_referees(self, detections):
        mask = detections.class_id == 3  # keep only referee
        return detections[mask]