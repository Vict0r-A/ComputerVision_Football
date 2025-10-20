
import supervision as sv
import pandas as pd
import numpy as np
import cv2
class ObjectTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        
    def update_tracks(self, detections):
        return self.tracker.update_with_detections(detections)


class BallTracking:
    """
    Collect ball bboxes for all frames, then interpolate in one shot.
    This matches the repo: df.interpolate().bfill() using future frames.
    """
    def __init__(self):
        self._bboxes = []  # list of [x1,y1,x2,y2] or NaNs

    def push(self, ball_detections):
        if len(ball_detections) > 0:
            self._bboxes.append(ball_detections.xyxy[0].tolist())
        else:
            self._bboxes.append([np.nan, np.nan, np.nan, np.nan])

    def finalise(self):
        if not self._bboxes:
            return []

        df = pd.DataFrame(self._bboxes, columns=["x1","y1","x2","y2"], dtype=float)
        df = df.interpolate().bfill()     #exactly like the repo
        self._bboxes = df.to_numpy().tolist()
        return self._bboxes  # list of [x1,y1,x2,y2] for each frame