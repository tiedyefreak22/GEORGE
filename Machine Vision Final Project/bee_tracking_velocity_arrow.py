
import cv2
import numpy as np
from collections import OrderedDict

class BeeTracker:
    def __init__(self, max_disappeared=3, max_distance=500):
        # max_dissappeared is threshold number of frames for dropping a track if no movement is detected.
        # max_distance is maximum allowable distance to match a new detection to an existing track. 
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.trajectories = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.trajectories[self.next_object_id] = [centroid]
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.trajectories[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, self.trajectories

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = np.linalg.norm(np.array(object_centroids)[:, None] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            assigned_rows = set()
            assigned_cols = set()
            for row, col in zip(rows, cols):
                if row in assigned_rows or col in assigned_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.trajectories[object_id].append(input_centroids[col])
                self.disappeared[object_id] = 0
                assigned_rows.add(row)
                assigned_cols.add(col)
            unused_rows = set(range(D.shape[0])) - assigned_rows
            unused_cols = set(range(D.shape[1])) - assigned_cols
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            for col in unused_cols:
                self.register(input_centroids[col])
        return self.objects, self.trajectories

def estimate_orientation(contour):
    if len(contour) < 5:
        return None
    data_pts = np.array(contour).reshape(-1, 2).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(data_pts, mean=None)
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
    return angle

def main():
    alpha = 0.15 # determines how quickly the background adapts to changes per equation: background = (1 - alpha) * background + alpha * current_frame.  Higher alpha adapts faster
    min_area = 1500 # min area for a contour
    tracker = BeeTracker()
    cap = cv2.VideoCapture("bees.mp4")
    background = None
    prev_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        if background is None:
            background = gray_blurred.copy().astype("float")
            prev_gray = gray_blurred.copy()
            continue
        cv2.accumulateWeighted(gray_blurred, background, alpha)
        background_uint8 = cv2.convertScaleAbs(background)
        diff = cv2.absdiff(gray_blurred, background_uint8)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append(np.array([cx, cy]))
        objects, trajectories = tracker.update(np.array(centroids))
        for object_id, centroid in objects.items():
            cv2.circle(frame, tuple(centroid), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {object_id}", (centroid[0] + 5, centroid[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            orientation = estimate_orientation(cnt)
            if orientation is not None:
                rect = cv2.boundingRect(cnt)
                x, y = rect[:2]
                cv2.putText(frame, f"{orientation:.1f}°", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 1)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_blurred, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        for object_id, centroid in objects.items():
            x, y = centroid
            if 0 <= y < flow.shape[0] and 0 <= x < flow.shape[1]:
                fx, fy = flow[y, x]
                if np.hypot(fx, fy) > 1:
                    end_point = (int(x + fx * 2.5), int(y + fy * 2.5)) # 2.5 corresponds to velocity arrow length
                    cv2.arrowedLine(frame, (x, y), end_point, (255, 0, 0), 2, tipLength=0.3)

        prev_gray = gray_blurred.copy()

        for object_id, points in trajectories.items():
            if len(points) < 2:
                continue
            for i in range(1, len(points)):
                if points[i - 1] is None or points[i] is None:
                    continue
                cv2.line(frame, tuple(points[i - 1]), tuple(points[i]), (0, 255, 255), 1)

        cv2.imshow("Bee Tracking", frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
