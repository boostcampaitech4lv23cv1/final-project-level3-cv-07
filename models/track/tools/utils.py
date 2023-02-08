import time
import sys
import glob

import cv2
from PIL import Image
from facenet_pytorch import MTCNN
from sklearn.cluster import DBSCAN

def extract_feature(target_path, work_dir):
    # make dir

    mtcnn = MTCNN(margin=30)
    img = cv2.imread(target_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_path = work_dir + "/target_detect.png"
    img_cropped = mtcnn(img, save_path=img_path)

    if img_cropped is None:
        print("Error: Your target image has no valid face tracking. Check again.")
        sys.exit(0)

def get_frame_num(source):
    cap = cv2.VideoCapture(source)
    i = 0
    while True:
        _, cur_frame = cap.read()
        if cur_frame is None:
            break
        i += 1

    return i

def bbox_scale_up(x_min, y_min, x_max, y_max, height, width, scale):
    h = y_max - y_min
    w = x_max - x_min
    x_min = int(max(0, x_min - w // scale))
    y_min = int(max(0, y_min - h // scale))
    x_max = int(min(width, x_max + w // scale))
    y_max = int(min(height, y_max + h // scale))
    return x_min, y_min, x_max, y_max

def dbscan(target_dir, tracklet_dir):
    tracklet_imgs = glob.glob(tracklet_dir + "/*.png")
    # encodings = [DeepFace.represent(img_path=img,enforce_detection=False,model_name="Facenet512") for img in tracklet_imgs]
    data = []
    for imagePath in tracklet_imgs:
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        d = [
            {"imagePath": imagePath, "loc": box, "encoding": enc}
            for (box, enc) in zip(boxes, encodings)
        ]
        data.extend(d)
    encodings = [d["encoding"] for d in data]
    # dump the facial encodings data to disk
    stime = time.time()
    clt = DBSCAN(metric="euclidean")
    clt.fit(encodings)
    etime = time.time()
    label_ids = np.unique(clt.labels_)
    numUniqueFaces = len(np.where(label_ids > -1)[0])
    if opt.verbose:
        print(f"DBSCAN time elapsed :{etime - stime}")
        print("[INFO] # unique faces: {}".format(numUniqueFaces))
