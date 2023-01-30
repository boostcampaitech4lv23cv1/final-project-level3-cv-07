import cv2

def set_vid_resolution(source_file, save_file, w, h):
    
    cap = cv2.VideoCapture(source_file)
    frame_array = []
    i = 0

    while True:
        ret, cur_frame = cap.read()
        if cur_frame is None:
            break
        image = image[:h, :w, :]
        frame_array.append(image)
        i += 1

    out = cv2.VideoWriter(
    save_file,
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (w,h),
    )

    for i in tqdm(range(len(frame_array))):
        out.write(frame_array[i])
    out.release()

    return None

if __name__ =="__main__":
    w = 1299 
    h = 1000
    source_file = "/opt/ml/final-project-level3-cv-07/database/uploaded_video/1080p.mp4"
    save_file = f"/opt/ml/final-project-level3-cv-07/database/uploaded_video/resized_{w}_{h}_1080p.mp4"
    set_vid_resolution(source_file, save_file, w, h)