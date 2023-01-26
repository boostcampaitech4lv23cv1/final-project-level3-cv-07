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

def createDirectory(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Failed to create the directory.")

def save_vid_2_img(vid_path, save_dir):
    cap = cv2.VideoCapture(vid_path)
    i = 0
    print("#####")
    print(vid_path)
    print(cap)
    print("#####")

    while(True): #cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv2.resize(frame, (1280, 720))
        # frame_list.append(frame)
        cv2.imwrite(save_dir + f'/frame_{i+1}.png', frame)
        i += 1

def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image
    
def cartoonize(load_folder, save_folder, model_path):
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
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
            image = resize_crop(image)
            batch_image = image.astype(np.float32)/127.5 - 1
            batch_image = np.expand_dims(batch_image, axis=0)
            output = sess.run(final_out, feed_dict={input_photo: batch_image})
            output = (np.squeeze(output)+1)*127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, output)
        except:
            print('cartoonize {} failed'.format(load_path))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='chim', help='name of video project')

    opt = parser.parse_args()

    model_path = '/opt/ml/final-project-level3-cv-07/models/track/cartoonize/saved_models'
    load_dir = '/opt/ml/final-project-level3-cv-07/models/track/cartoonize/' + opt.project + '/image_orig'
    save_dir = '/opt/ml/final-project-level3-cv-07/models/track/cartoonize/' + opt.project + '/image_cart'
    input_video = '/opt/ml/final-project-level3-cv-07/models/track/assets/' + opt.project + '.mp4'
    
    
    createDirectory(load_dir)
    createDirectory(save_dir)

    save_vid_2_img(input_video,load_dir)

    s = time.time()
    cartoonize(load_dir, save_dir, model_path)
    e = time.time()
    print(f"Total elapsed time: {e-s}")