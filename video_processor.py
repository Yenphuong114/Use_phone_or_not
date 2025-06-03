import cv2
import numpy as np
import os

# Khởi tạo danh sách điểm để vẽ tứ giác
points = []

# Hàm callback để xử lý sự kiện click chuột
def click_event(event, x, y, flags, params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(frame, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(frame, tuple(points[0]), tuple(points[3]), (0, 255, 0), 2)
            cv2.imshow('Frame', frame)

# Đọc video
video_path = 'vid_data/Person_1.mp4'
cap = cv2.VideoCapture(video_path)

# Tạo thư mục để lưu frame
output_dir = 'data/not_use_phone_P1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Đọc frame đầu tiên
ret, frame = cap.read()
if not ret:
    print('Không thể đọc video')
    exit()

# Hiển thị frame và thiết lập callback
cv2.imshow('Frame', frame)
cv2.setMouseCallback('Frame', click_event)

frame_count = 0
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p') and len(points) == 4:
        # Chuyển đổi points thành numpy array
        pts = np.array(points)
        
        # Reset video về frame đầu
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Cắt vùng được chọn
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Lưu frame
            output_path = os.path.join(output_dir, f'frame_{frame_idx}.jpg')
            cv2.imwrite(output_path, masked_frame)
            frame_idx += 1
            
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()