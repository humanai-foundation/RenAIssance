# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc 

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    
    # Use sets for faster, cleaner lookups
    valid_img_exts = {'.jpg', '.jpeg', '.gif', '.png', '.pgm', '.tiff', '.tif'}
    valid_mask_exts = {'.bmp'}
    valid_gt_exts = {'.xml', '.gt', '.txt'}

    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = ext.lower()
            
            if ext in valid_img_exts:
                img_files.append(os.path.join(dirpath, file))
            elif ext in valid_mask_exts:
                mask_files.append(os.path.join(dirpath, file))
            elif ext in valid_gt_exts:
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
                
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
    """ save text detection result one by one
    Args:
        img_file (str): image file name
        img (array): raw image context
        boxes (array): array of result file
            Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
    Return:
        None
    """
    img = np.array(img)

    # make result file list
    filename, _ = os.path.splitext(os.path.basename(img_file))

    # Robust directory creation avoiding race conditions and handling nested paths
    os.makedirs(dirname, exist_ok=True)

    # Safe path concatenation using os.path.join
    res_file = os.path.join(dirname, f"res_{filename}.txt")
    res_img_file = os.path.join(dirname, f"res_{filename}.jpg")

    with open(res_file, 'w') as f:
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            strResult = ','.join([str(p) for p in poly]) + '\n'
            f.write(strResult)

            poly = poly.reshape(-1, 2)
            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            
            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                
                x, y = int(poly[0][0]), int(poly[0][1])
                cv2.putText(img, str(texts[i]), (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness=1)
                cv2.putText(img, str(texts[i]), (x, y), font, font_scale, (0, 255, 255), thickness=1)

    # Save result image
    cv2.imwrite(res_img_file, img)
