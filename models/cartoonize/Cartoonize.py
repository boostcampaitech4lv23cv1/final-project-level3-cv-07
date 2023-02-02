import os
import cv2
import numpy as np
import tensorflow as tf
import network
import guided_filter
from tqdm import tqdm
import time
import cv2
import argparse

def save_vid_2_img(vid_path, save_dir):
    cap = cv2.VideoCapture(vid_path)
    i = 0
    print("vid_path", vid_path)
    print("cap", cap)

    while True:  # cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv2.resize(frame, (1280, 720))
        # frame_list.append(frame)
        frame = resize_crop_orig(frame)
        cv2.imwrite(save_dir + f"/frame_{i+1}.png", frame)
        i += 1


def resize_crop_cart(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def resize_crop_orig(image):
    h, w, c = np.shape(image)
    if min(h, w) > 1080:
        if h > w:
            h, w = int(1080 * h / w), 1080
        else:
            h, w = 1080, int(1080 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def cartoonize(load_folder, save_folder, model_path):
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if "generator" in var.name]
    saver = tf.train.Saver(var_list=gene_vars)

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction=0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    name_list = os.listdir(load_folder)
    for name in tqdm(name_list):
        try:
            load_path = os.path.join(load_folder, name)
            save_path = os.path.join(save_folder, name)
            image = cv2.imread(load_path)
            image = resize_crop_cart(image)
            batch_image = image.astype(np.float32) / 127.5 - 1
            batch_image = np.expand_dims(batch_image, axis=0)
            output = sess.run(final_out, feed_dict={input_photo: batch_image})
            output = (np.squeeze(output) + 1) * 127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, output)
        except:
            print("cartoonize {} failed".format(load_path))


if __name__ == "__main__":
    from pymongo import MongoClient

    client = MongoClient()

    db = client["cafe"]
    collection = db['env']

    base_info = collection.find_one({'name': 'base'})
    database_info = collection.find_one({'name': 'database'})
    backend_info = collection.find_one({'name': 'backend'})
    cartoonize_info = collection.find_one({'name': 'cartoonize'})
    track_info = collection.find_one({'name': 'track'})
    
    class Opt:
        weights= f"{track_info['dir']}/pretrained/yolov7-tiny.pt"
        source = f"{database_info['dir']}/uploaded_video/video.mp4"
        target = f"{database_info['dir']}/target/target.jpeg"
        img_size = 1920
        conf_thres = 0.09
        iou_thres = 0.7
        sim_thres = 0.35
        device = "0"
        nosave = None
        classes = None
        agnostic_nms = True
        augment = None
        update = None
        work_dir= f"{database_info['dir']}/work_dir"
        name = "exp"
        exist_ok = None
        save_results = True
        save_txt_tidl = None
        kpt_label = 5
        hide_conf = (False,)
        line_thickness = 3
        save_img = True
        
        # Tracking args
        track_high_thresh = 0.3
        track_low_thresh = 0.05
        new_track_thresh = 0.4
        track_buffer = 30
        match_thresh = 0.7
        conf_thresh = 0.7  # added
        aspect_ratio_thresh = 1.6
        min_box_area = 10
        min_frame = 5  # added
        dbscan = False  # added
        mot20 = True
        save_crop = None

        # CMC
        cmc_method = "sparseOptFlow"
        swap_all_face = False 
        verbose = False
        # ReID
        with_reid = False
        fast_reid_config = r"fast_reid/configs/MOT17/sbs_S50.yml"
        fast_reid_weights = r"pretrained/mot17_sbs_S50.pth"
        proximity_thresh = 0.5
        appearance_thresh = 0.25
        jde = False
        ablation = False
    
    opt = Opt()
    
    model_path =f"{cartoonize_info['dir']}/saved_models"
    load_dir = f"{opt.work_dir}/image_orig"
    save_dir = f"{opt.work_dir}/image_cart"
    input_video = f"{database_info['dir']}/uploaded_video/video.mp4"
    
    
    save_vid_2_img(input_video, load_dir)

    s = time.time()
    cartoonize(load_dir, save_dir, model_path)
    e = time.time()
    print(f"Total elapsed time: {e-s}")
