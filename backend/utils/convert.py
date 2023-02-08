import cv2
import os
import numpy as np

from .mask_generator import (
    mask_generator_v0,
    mask_generator_v1,
    mask_generator_v2,
    mask_generator_v3,
)

# from deepface import DeepFace


def parsing_results(track_info, valid_ids, num_frames, swap_all_face):
    parsed_lines = []

    # Remove unnecessary info and casting data type (str -> int)
    for line in track_info:
        line = list(line.values())
        line = list(map(int, list(map(float, line))))
        parsed_lines.append(line)

    # Remove redundant info and soted by frame > obj_id
    parsed_lines = sorted(
        list(map(list, set(list(map(tuple, parsed_lines))))),
        key=lambda x: (x[0], x[1]),
    )

    # Summary info (per frame)
    final_lines = []
    i = 1
    for line in parsed_lines:
        frame, obj_id, x, y, w, h, conf = line

        # save all face (for debugging)
        if swap_all_face:
            if not final_lines or frame != final_lines[-1][0]:
                final_lines.append([frame, x, y, x + w, y + h])
            else:
                final_lines[-1] = final_lines[-1] + [x, y, x + w, y + h]

        # save valid face
        else:
            if obj_id in valid_ids:
                if not final_lines or frame != final_lines[-1][0]:
                    final_lines.append([frame, x, y, x + w, y + h])
                else:
                    final_lines[-1] = final_lines[-1] + [x, y, x + w, y + h]

    total_lines = []

    idx = 1
    for i in range(len(final_lines)):
        if idx < final_lines[i][0]:
            while idx < final_lines[i][0]:
                total_lines.append([idx + 1])
                idx += 1
        total_lines.append(final_lines[i])
        idx += 1

    while len(total_lines) < num_frames:
        total_lines.append([total_lines[-1][0] + 1])

    return total_lines


def bbox_scale_up(x_min, y_min, x_max, y_max, height, width, scale):
    h = y_max - y_min
    w = x_max - x_min
    x_min = int(max(0, x_min - w // scale))
    y_min = int(max(0, y_min - h // scale))
    x_max = int(min(width, x_max + w // scale))
    y_max = int(min(height, y_max + h // scale))
    return x_min, y_min, x_max, y_max


def save_face_swapped_vid(final_lines, work_dir, fps):
    import time
    from tqdm import tqdm

    img = cv2.imread(f"{work_dir}/image_orig/frame_1.png")
    height, width, layers = img.shape
    size = (width, height)
    swap_s = time.time()

    frame_array = []
    # face swap per frame
    for line in tqdm(final_lines):
        assert (len(line) - 1) % 4 == 0
        frame_idx = line[0]  # Image Index starts from 1
        orig_img = cv2.imread(f"{work_dir}/image_orig/frame_{frame_idx}.png")
        cart_img = cv2.imread(f"{work_dir}/image_cart/frame_{frame_idx}.png")
        resized_cart_img = cv2.resize(cart_img, size, interpolation=cv2.INTER_LINEAR)
        face_swapped_img = orig_img
        for i in range(((len(line) - 1) // 4)):
            x_min, y_min, x_max, y_max = (
                line[4 * i + 1],
                line[4 * i + 2],
                line[4 * i + 3],
                line[4 * i + 4],
            )  # original bbox
            sx_min, sy_min, sx_max, sy_max = bbox_scale_up(
                x_min, y_min, x_max, y_max, height, width, 2
            )  # scaled bbox ('s' means scaled)

            """
            Select mask generator function
            - mask_generator_v0: same as not using mask
            - mask_generator_v1: using Euclidean distance (L2 distance) and thresholding
            - mask_generator_v2: using Manhattan distance (L1 distance) and thresholding
            - mask_generator_v3: using padding
            """

            mask, inv_mask = mask_generator_v3(sx_min, sy_min, sx_max, sy_max)
            orig_face = orig_img[sy_min:sy_max, sx_min:sx_max]
            cart_face = resized_cart_img[sy_min:sy_max, sx_min:sx_max]
            swap_face = np.multiply(cart_face, mask) + np.multiply(orig_face, inv_mask)
            face_swapped_img[sy_min:sy_max, sx_min:sx_max] = swap_face
        frame_array.append(face_swapped_img)
    swap_e = time.time()
    print(f"Time Elapsed for face swap: {swap_e - swap_s}")

    out = cv2.VideoWriter(
        os.path.join(work_dir, "result.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    for i in tqdm(range(len(frame_array))):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
