 
import cv2, os, numpy as np, supervision as sv
from object_detection.detection import ObjectDetection
from object_tracking.tracking import ObjectTracker, BallTracking
from object_tracking.annotations import PlayerRefereeAnnotation, BallAnnotation

def main(video_path="input_videos/test1.mp4", output_path="tracked_output_1.mp4"):
    os.makedirs("output_videos", exist_ok=True)
    
    #make a directory, exist ok default false

    output_video = os.path.join("output_videos", output_path) #join components of a path, takes multiple rgumentts , returns a string

    # initialise the instances of these classes 
    object_detector = ObjectDetection()
    player_tracker = ObjectTracker()
    ball_tracker = BallTracking()

    # read all frames first
    cap = cv2.VideoCapture(video_path) #opens the video, creating an object, decoding it decompressing it
#         
    frames = []
    while True:
        ret, frame = cap.read() 
        #read a frame from a video returns a tuple  boolean true or fals, andthe frame as a numpy arrray

        if not ret:
            break
        frames.append(frame)
    cap.release()

    # storage for per-frame tracks
    tracked_players_per_frame = []
    referees_per_frame = []
 
    for frame in frames:
        detections = object_detector.detect(frame)
        players   = object_detector.filter_players(detections)
        ball      = object_detector.filter_ball(detections)
        referees  = object_detector.filter_referees(detections)

        tracked_players = player_tracker.update_tracks(players)

        tracked_players_per_frame.append(tracked_players)
        referees_per_frame.append(referees)

        # collect ball bbox for this frame (may be NaN)
        ball_tracker.push(ball)

    # interpolate ball across ALL frames (uses future frames)
    ball_bboxes = ball_tracker.finalise()  # list aligned to frames

  
    #Codec is a program/algorithm that compresses raw video data into a file format
    h, w = frames[0].shape[:2]
    #FourCC = Four Character Code, string telling which. codec to use
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")   #*unpacks the string
    out = cv2.VideoWriter(output_video, fourcc, 30, (w, h)) #write frames to create a new video file by creating this object
    #videowriter uses the codec to compress the frame and append it to the mp4 file
    #save it to output_video, using codec, 30 fps, frame size
    for i, frame in enumerate(frames):
        tracked_players = tracked_players_per_frame[i]
        referees        = referees_per_frame[i]
        ball_bbox       = ball_bboxes[i]

        # Players ring and ID
        for bbox, tid in zip(tracked_players.xyxy, tracked_players.tracker_id):
            frame = PlayerRefereeAnnotation.draw_ring(frame, bbox, color=(0,255,0), track_id=tid)

        # Referees ring and label
        for bbox in referees.xyxy:
            frame = PlayerRefereeAnnotation.draw_ring(frame, bbox, color=(0,0,255), label="Referee")

        # Ball downward triangle 
        if len(ball_bbox) == 4 and not np.isnan(ball_bbox).any():
            frame = BallAnnotation.draw_triangle(frame, ball_bbox, color=(0,255,255))

        out.write(frame) #adds the encoded compressed frame to the file

    out.release() #close the video source. stop reading frames
    cv2.destroyAllWindows() #closes all OpenCV-created windows. equivalent to clicking x to close

if __name__ == "__main__":
    print("running")
    main()