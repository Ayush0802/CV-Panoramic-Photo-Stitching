import os
import cv2

FEATURE_THRESHOLD = 0.01
DESCRIPTOR_SIZE   = 5
MATCHING_Y_RANGE  = 30
RANSAC_K              = 1000
RANSAC_THRES_DISTANCE = 3
BLEND_WIDTH = 30

def load_images(source_dir):
    imgs = []
    
    for (dirpath, dirnames, filenames) in os.walk(source_dir):
        imgs.extend([filename for filename in filenames if filename.endswith('.jpg') or filename.endswith('.png')])
        break
    
    image_list = [cv2.imread(os.path.join(source_dir, img), 1) for img in imgs]

    return image_list

def parse(source_dir):
    filenames = []
    focal_length = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, f, *rest) = line.split()
        filenames += [filename]
        focal_length += [float(f)]
    
    img_list = [cv2.imread(os.path.join(source_dir, f), 1) for f in filenames]

    return (img_list, focal_length)


