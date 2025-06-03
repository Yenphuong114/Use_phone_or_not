import cv2
import numpy as np
import os
import mediapipe as mp

class VideoProcessor:
    def __init__(self):
        self.points = []
        self.rectangles = []
        self.current_frame = None
        self.video_path = None
        self.drawing = False
        self.scale_factor = 0.25  # Scale down video by 75% for better viewing
        self.point_size = 8  # Increased point size
        self.line_thickness = 3  # Increased line thickness
        self.cap = None
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        # Initialize MediaPipe Pose Detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5)

    def mouse_callback(self, event, x, y, flags, param):
        # Convert mouse coordinates back to original scale
        x = int(x / self.scale_factor)
        y = int(y / self.scale_factor)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                display_frame = self.current_frame.copy()
                display_frame = cv2.resize(display_frame, None, fx=self.scale_factor, fy=self.scale_factor)
                
                # Draw scaled points with larger size
                for point in self.points:
                    scaled_point = (int(point[0] * self.scale_factor), int(point[1] * self.scale_factor))
                    cv2.circle(display_frame, scaled_point, self.point_size, (255, 255, 255), -1)
                    cv2.circle(display_frame, scaled_point, self.point_size, (0, 0, 255), 2)
                if len(self.points) > 1:
                    for i in range(len(self.points)-1):
                        pt1 = (int(self.points[i][0] * self.scale_factor), int(self.points[i][1] * self.scale_factor))
                        pt2 = (int(self.points[i+1][0] * self.scale_factor), int(self.points[i+1][1] * self.scale_factor))
                        cv2.line(display_frame, pt1, pt2, (0, 255, 0), self.line_thickness)
                cv2.imshow('Frame', display_frame)
                
                if len(self.points) == 4:
                    display_frame = self.current_frame.copy()
                    display_frame = cv2.resize(display_frame, None, fx=self.scale_factor, fy=self.scale_factor)
                    for point in self.points:
                        scaled_point = (int(point[0] * self.scale_factor), int(point[1] * self.scale_factor))
                        cv2.circle(display_frame, scaled_point, self.point_size, (0, 0, 255), -1)
                    for i in range(len(self.points)):
                        pt1 = (int(self.points[i][0] * self.scale_factor), int(self.points[i][1] * self.scale_factor))
                        pt2 = (int(self.points[(i+1)%4][0] * self.scale_factor), int(self.points[(i+1)%4][1] * self.scale_factor))
                        cv2.line(display_frame, pt1, pt2, (0, 255, 0), self.line_thickness)
                    cv2.imshow('Frame', display_frame)
                    
                    self.rectangles.append(self.points)
                    self.points = []
            else:
                self.points = [(x, y)]
                display_frame = self.current_frame.copy()
                display_frame = cv2.resize(display_frame, None, fx=self.scale_factor, fy=self.scale_factor)
                scaled_point = (int(x * self.scale_factor), int(y * self.scale_factor))
                cv2.circle(display_frame, scaled_point, self.point_size, (255, 255, 255), -1)
                cv2.circle(display_frame, scaled_point, self.point_size, (0, 0, 255), 2)
                cv2.imshow('Frame', display_frame)

    def process_video(self):
        self.video_path = "vid_data/Person_4.mp4"
        
        if not os.path.exists('data/use_phone/Person(3)'):
            os.makedirs('data/use_phone/Person(3)', exist_ok=True)

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return

        ret, self.current_frame = self.cap.read()
        if not ret:
            print("Error: Could not read video.")
            return

        # Get video dimensions and calculate scaled dimensions
        frame_height = int(self.current_frame.shape[0] * self.scale_factor)
        frame_width = int(self.current_frame.shape[1] * self.scale_factor)

        # Create resizable window and set to scaled video size
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame', frame_width, frame_height)
        cv2.setMouseCallback('Frame', self.mouse_callback)

        frame_count = 0
        processing = False

        while True:
            if not processing:
                display_frame = self.current_frame.copy()
                display_frame = cv2.resize(display_frame, None, fx=self.scale_factor, fy=self.scale_factor)
                
                for rect_points in self.rectangles:
                    for point in rect_points:
                        scaled_point = (int(point[0] * self.scale_factor), int(point[1] * self.scale_factor))
                        cv2.circle(display_frame, scaled_point, self.point_size, (255, 255, 255), -1)
                        cv2.circle(display_frame, scaled_point, self.point_size, (0, 0, 255), 2)
                    for i in range(len(rect_points)):
                        pt1 = (int(rect_points[i][0] * self.scale_factor), int(rect_points[i][1] * self.scale_factor))
                        pt2 = (int(rect_points[(i+1)%4][0] * self.scale_factor), int(rect_points[(i+1)%4][1] * self.scale_factor))
                        cv2.line(display_frame, pt1, pt2, (0, 255, 0), self.line_thickness)
                
                for point in self.points:
                    scaled_point = (int(point[0] * self.scale_factor), int(point[1] * self.scale_factor))
                    cv2.circle(display_frame, scaled_point, self.point_size, (255, 255, 255), -1)
                    cv2.circle(display_frame, scaled_point, self.point_size, (0, 0, 255), 2)
                if len(self.points) > 1:
                    for i in range(len(self.points)-1):
                        pt1 = (int(self.points[i][0] * self.scale_factor), int(self.points[i][1] * self.scale_factor))
                        pt2 = (int(self.points[i+1][0] * self.scale_factor), int(self.points[i+1][1] * self.scale_factor))
                        cv2.line(display_frame, pt1, pt2, (0, 255, 0), self.line_thickness)
                cv2.imshow('Frame', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.points = []
                ret, self.current_frame = self.cap.read()
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.rectangles = []
            elif key == ord('p'):
                processing = True
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    # Process every frame
                    for idx, rect_points in enumerate(self.rectangles):
                        x_coords = [p[0] for p in rect_points]
                        y_coords = [p[1] for p in rect_points]
                        x1, x2 = min(x_coords), max(x_coords)
                        y1, y2 = min(y_coords), max(y_coords)

                        cropped = frame[y1:y2, x1:x2]
                        if cropped.size > 0:
                            # Save all frames from selected regions
                            if not os.path.exists('data/use_phone/Person(3)'):
                                os.makedirs('data/use_phone/Person(3)', exist_ok=True)
                            cv2.imwrite(f'data/use_phone/Person(3)/frame_{frame_count}_region_{idx}.jpg', cropped)

                    frame_count += 1
                break

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.process_video()