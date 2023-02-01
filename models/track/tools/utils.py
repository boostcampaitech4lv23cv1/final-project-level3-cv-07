import os
import cv2


def createDirectory(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Failed to create the directory.")


def get_frame_num(source):
    cap = cv2.VideoCapture(source)
    frame_list = []
    i = 0
    while True:
        ret, cur_frame = cap.read()
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


def calc_similarity_v1(target_feature, tracker_feat, sim_thres):
    print("Similairties(cosine) list: ")
    print(
        cdist(
            target_feature.reshape(1, target_feature.size),
            list(tracker_feat.values()),
            metric="cosine",
        )
    )
    print("Similairties(Euclidean) list: ")
    print(
        cdist(
            target_feature.reshape(1, target_feature.size),
            list(tracker_feat.values()),
            metric="euclidean",
        )
    )
    print(f"Similarity Threshold : {opt.sim_thres}")
    sim = (
        cdist(
            target_feature.reshape(1, target_feature.size),
            list(tracker_feat.values()),
            metric="cosine",
        )
        > sim_thres
    )  # distance가 1 이상인 (즉, 비슷하지 않은) tracker 찾기
    t_ids = np.asarray(list(tracker_feat.keys()))
    valid_ids = t_ids[sim[0]]  # key에 넣어서 해당 tracker ID만을 뽑아내기
    return valid_ids

def calc_similarity_v2(dfs, t_ids, tracklet_dir):

    targeted_ids = {}

    for i in range(len(dfs)):
        id, sim = (
            int(dfs.iloc[i].identity.split("/")[-1].split(".")[0]),
            dfs.iloc[i]["VGG-Face_cosine"],
        )
        targeted_ids[id] = sim

    targeted_ids = dict(sorted(targeted_ids.items(), key=lambda x: x[1]))
    # t_ids = dict(sorted(t_ids.items(),key=lambda x : x[1],reverse=True))

    best_matched_id = list(targeted_ids.keys())[0]

    second_dfs = DeepFace.find(
        img_path=f"{tracklet_dir}/{best_matched_id}.png",
        db_path=tracklet_dir,
        enforce_detection=False,
        model_name="VGG-Face",
    )
    targeted_ids = {}
    for i in range(len(second_dfs)):
        id, sim = (
            int(second_dfs.iloc[i].identity.split("/")[-1].split(".")[0]),
            second_dfs.iloc[i]["VGG-Face_cosine"],
        )
        if sim < 0.25:
            id_conf = t_ids.pop(id)
            targeted_ids[id] = (id_conf, sim)

    return targeted_ids, t_ids


def calc_similarity_v3(dfs, t_ids, tracklet_dir):
    sim_dict, sim_cnt = defaultdict(float), defaultdict(int)
    best_matched_id = int(dfs.identity[0].split("/")[-1].split(".")[0])

    second_dfs = DeepFace.find(
        img_path=f"{tracklet_dir}/{best_matched_id}.png",
        db_path=tracklet_dir,
        enforce_detection=False,
        model_name="VGG-Face",
    )
    cnt = 0
    for i in range(len(second_dfs)):
        second_id, second_sim = (
            int(second_dfs.identity[i].split("/")[-1].split(".")[0]),
            second_dfs["VGG-Face_cosine"][i],
        )
        if second_sim < 0.25:
            cnt += 1
            third_dfs = DeepFace.find(
                img_path=f"{tracklet_dir}/{second_id}.png",
                db_path=tracklet_dir,
                enforce_detection=False,
                model_name="VGG-Face",
            )
            for i in range(len(third_dfs)):
                third_id, third_sim = (
                    int(third_dfs.identity[i].split("/")[-1].split(".")[0]),
                    third_dfs["VGG-Face_cosine"][i],
                )
                if third_sim < 0.25:
                    sim_dict[third_id] += third_sim
                    sim_cnt[third_id] += 1
    for id in sim_dict.keys():
        sim_dict[id] /= sim_cnt[id]

    targeted_ids = {}

    if sim_cnt:
        for tid, t_cnt in sim_cnt.items():
            if t_cnt / cnt >= 0.75:
                targeted_ids[tid] = (t_ids.pop(tid), sim_dict[tid])

    return targeted_ids, t_ids


def write_results(filename, results):
    with open(filename, "a", encoding="UTF-8") as f:
        f.writelines(results)


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
