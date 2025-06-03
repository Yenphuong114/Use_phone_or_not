import cv2
import os
import numpy as np
from pathlib import Path

def create_videos_from_frames():
    # Đường dẫn thư mục chứa frames và output
    input_dir = 'data/use_phone/Person(3)'
    output_dir = 'vid_frame_use_phone/vid_use_phone_P3'
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy danh sách tất cả các frame
    frames = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    frames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sắp xếp theo số frame
    
    # Thông số video
    fps = 30
    frames_per_video = fps * 5  # 5 giây mỗi video
    
    # Đọc frame đầu tiên để lấy kích thước
    first_frame = cv2.imread(os.path.join(input_dir, frames[0]))
    height, width = first_frame.shape[:2]
    
    # Tạo các video 5 giây
    for i in range(0, len(frames), frames_per_video):
        video_frames = frames[i:i + frames_per_video]
        
        if len(video_frames) < frames_per_video:
            # Nếu không đủ frame cho 5 giây, bỏ qua
            break
        
        # Tạo VideoWriter
        output_path = os.path.join(output_dir, f'video_{i//frames_per_video}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Buffer để xử lý frame
        frame_buffer = []
        
        # Đọc và xử lý từng frame
        for frame_name in video_frames:
            frame = cv2.imread(os.path.join(input_dir, frame_name))
            
            if frame is not None:
                frame_buffer.append(frame)
                
                # Khi buffer đủ lớn, xử lý và ghi frame
                if len(frame_buffer) >= 2:
                    # Nội suy frame để tạo chuyển động mượt mà
                    for j in range(len(frame_buffer) - 1):
                        current = frame_buffer[j]
                        next_frame = frame_buffer[j + 1]
                        
                        # Ghi frame hiện tại
                        out.write(current)
                        
                        # Tạo frame trung gian để làm mượt chuyển động
                        if j < len(frame_buffer) - 2:
                            interpolated = cv2.addWeighted(current, 0.5, next_frame, 0.5, 0)
                            out.write(interpolated)
                    
                    # Giữ lại frame cuối cùng trong buffer
                    frame_buffer = [frame_buffer[-1]]
        
        # Ghi frame cuối cùng
        if frame_buffer:
            out.write(frame_buffer[0])
        
        # Giải phóng VideoWriter
        out.release()
        
        print(f'Đã tạo video: {output_path}')

if __name__ == '__main__':
    create_videos_from_frames()