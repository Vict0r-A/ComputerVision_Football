# from ultralytics import YOLO
# import cv2

# # load your model (replace with your trained model path if custom)
# model = YOLO("models/keypoint.pt")  # or "best.pt" if you trained one

# # input video
# video_path = "input_videos/test1.mp4"
# cap = cv2.VideoCapture(video_path)

# # output video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = cap.get(cv2.CAP_PROP_FPS)
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter("output_custom.mp4", fourcc, fps, (w, h))

# frame_id = 0

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame, conf=0.5)  # run inference

#     annotated_frame = results[0].plot()  # draws boxes + keypoints
#     out.write(annotated_frame)

#     # OPTIONAL: print/save keypoints
#     for r in results:
#         if r.keypoints is not None:
#             print(f"Frame {frame_id} keypoints:", r.keypoints.xy.cpu().numpy())

#     frame_id += 1

# cap.release()
# out.release()
# print("Done! Output saved as output.mp4")




from ultralytics import YOLO

# Load your model
model = YOLO("models/pose.pt")

# Run inference on an image
results = model("input_videos/test2.mp4")

# Show results
for r in results:
    r.show()   # show in window
    r.save()   # save to runs/detect/predict