import cv2
import numpy as np

class PlayerRefereeAnnotation:
 
    def draw_ring(frame, bbox, color, track_id=None, label=None):
        """
        Draws a halo ellipse + optional ID/label box under a bounding box.
        """
        y2 = int(bbox[3])
        x_center = int((bbox[0] + bbox[2]) / 2)
        width = int(bbox[2] - bbox[0])

        # Halo ellipse
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Optional label box
        if track_id is not None or label is not None:
            rectangle_width = 60
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = y2 + 15
            y2_rect = y1_rect + rectangle_height

            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)

            text = str(track_id) if track_id is not None else str(label)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_x = x1_rect + (rectangle_width - tw) // 2
            text_y = y1_rect + (rectangle_height + th) // 2 - 3

            cv2.putText(frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

class BallAnnotation:
 
    def draw_triangle(frame, bbox, color=(0, 255, 255)):
        """
        Draws a downward yellow triangle with its tip at the top of the ball bbox.
        """
        x1, y1, x2, y2 = map(int, bbox)
        if any(v is None or v != v for v in [x1, y1, x2, y2]):  # handle NaN
            return frame

        x_center = int((x1 + x2) / 2)

        # Triangle points (tip exactly on bbox top)
        pts = np.array([
            [x_center, y1],            # tip (on top of bbox)
            [x_center - 12, y1 - 20],  # top-left
            [x_center + 12, y1 - 20]   # top-right
        ], np.int32)

        cv2.fillPoly(frame, [pts], color)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 0), thickness=2)

        return frame