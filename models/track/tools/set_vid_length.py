import cv2

def set_vid_length(source_file, save_file, num_frames):
    
    cap = cv2.VideoCapture(source_file)
    frame_array = []
    i = 0

    while i < num_frames:
        ret, cur_frame = cap.read()
        if cur_frame is None:
            break

        frame_array.append(cur_frame)
        i += 1
    
    height, width, layers = frame_array[0].shape
    size = (width, height)

    out = cv2.VideoWriter(
    save_file,
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    size,
    )

    for i in tqdm(range(len(frame_array))):
        out.write(frame_array[i])
    out.release()

    return None

if __name__ =="__main__":
    source_file = "/opt/ml/final-project-level3-cv-07/models/track/assets/4K_2.mp4"
    save_file = "/opt/ml/final-project-level3-cv-07/models/track/assets/4K_2_10s.mp4"
    set_vid_length(source_file, save_file, 600)