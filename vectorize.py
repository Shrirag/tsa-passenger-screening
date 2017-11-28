import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import csv, cv2, io, os
import numpy as np

##############################################################################
# Used to turn dataset of 'aps' files and their labels into a set of vectors
# grouped by body zones. Each vector contains all the releveant image crops as
# well as the training label for the body zone
##############################################################################

# Reads 'aps' file and returns as numpy array
def read_data(path):
    fid = open(path, 'r')
    scale = np.fromfile(fid, dtype = np.uint16, count = 1)
    fid.seek(512)
    data = np.fromfile(fid, dtype = np.uint16)
    return data.reshape(512, 660, 16, order='F')

# Takes 'aps' file path as input and returns split file from 3D image into 16 2D images
def aps_splitter(path):
	img = read_data(path)
	x, y, z = img.shape
	img_set = np.split(img, z, axis=(len(img.shape)-1))
	img_set = [np.rot90(pic.reshape(x, y)) for pic in img_set]
	img_set = [(img/256).astype(np.uint8) for img in img_set]
	img_set = [cv2.Canny(img, 0, 100) for img in img_set]
	img_set = [cv2.resize(img, (0, 0), fx=.0625, fy=.0625) for img in img_set]
	return img_set

# Takes a set of scans as returned from 'aps_splitter' and plots them
def plot_img_set(img_set):
    figarr, axarr = plt.subplots(4, 4, figsize=(20, 20))
    img_count = 0
    for i in range(4):
        for j in range(4): 
            if img_set[img_count] is not None: axarr[i][j].imshow(img_set[img_count])
            axarr[i][j].axis('off')
            img_count += 1
    plt.show()
    
# plots single image
def plot_img(img):
    figarr, axarr = plt.subplots(1, 1, figsize=(5, 5))
    axarr.imshow(img)
    axarr.axis('off')
    plt.show()

# Takes image scan and returns the crops of the image
# for each body zone (in terms of front facing image)
def sector_crops(img):

	return [
	    img[10:15, 0:12],
	    img[0:12, 0:10],
	    img[10:15, 20:32],
	    img[0:12, 21:32],
	    img[13:18, 0:32],
	    img[18:22, 0:16],
	    img[18:22, 16:32],
	    img[23:28, 0:14],
	    img[23:28, 14:17],
	    img[23:28, 17:32],
	    img[28:32, 0:16],
	    img[28:32, 16:32],
	    img[32:37, 0:16],
	    img[32:37, 16:32],
	    img[37:41, 0:16],
	    img[37:41, 16:32],
	    img[13:18, 0:32]
	]

# Takes a set of scans, crops each and groups all
# the cropped images by body zone
def group_by_zones(img_set):

    crops = []
    for img in img_set: crops.append(sector_crops(img))

    return  [ 

        [crops[0][0], crops[1][0], crops[2][0], None, None, None, crops[6][2], crops[7][2], 
         crops[8][2], crops[9][2], crops[10][2], None, None, crops[13][0], crops[14][0], crops[15][0]],

        [crops[0][1], crops[1][1], crops[2][1], None, None, None, crops[6][3], crops[7][3], 
         crops[8][3], crops[9][3], crops[10][3], None, None, crops[13][1], crops[14][1], crops[15][1]],

        [crops[0][2], crops[1][2], crops[2][2], crops[3][2], None, None, crops[6][0], crops[7][0], 
         crops[8][0], crops[9][0], crops[10][0], crops[11][0], None, None, crops[14][2], crops[15][2] ],

        [crops[0][3], crops[1][3], crops[2][3], crops[3][3], None, None, crops[6][1], crops[7][1], 
         crops[8][1], crops[9][1], crops[10][1], crops[11][1], None, None, crops[14][3], crops[15][3]],

        [crops[0][4], crops[1][4], crops[2][4], crops[3][4], crops[4][4], crops[5][4], crops[6][4], crops[7][4], 
         None, None, None, None, None, None, None, None],

        [crops[0][5], None, None, None, None, None, None, None, 
         crops[8][6], crops[9][6], crops[0][5], crops[11][5], crops[12][5], crops[13][5], crops[14][5], crops[15][5]],

        [crops[0][6], crops[1][6], crops[2][6], crops[3][6], crops[4][6], crops[5][6], crops[6][6], crops[7][6], 
         None, None, None, None, None, None, None, None],

        [crops[0][7], crops[1][7], None, None, None, None, None, crops[7][9], 
         crops[8][9], crops[9][9], crops[0][9], crops[11][9], crops[12][7], crops[13][7], crops[14][7], crops[15][7]],

        [crops[0][8], crops[1][8], crops[2][7], crops[3][7], crops[4][7], None, None, None, 
         crops[8][8], crops[9][8], None, None, None, None, crops[14][9], crops[15][8]],

        [crops[0][9], crops[1][9], crops[2][9], crops[3][9], crops[4][9], crops[5][7], crops[6][9], None, 
         None, None, None, None, None, None, None, crops[15][9]],

        [crops[0][10], crops[1][10], crops[2][10], crops[3][10], None, None, crops[6][11], crops[7][11], 
         crops[8][11], crops[9][11], crops[10][11], None, crops[12][10], crops[13][10], crops[14][10], crops[15][10]],

        [crops[0][11], crops[1][11], crops[2][11], crops[3][11], crops[4][11], crops[5][11], crops[6][11], crops[7][11], 
         crops[8][11], crops[9][11], crops[10][11], None, None, crops[13][11], crops[14][11], crops[15][11]],

        [crops[0][12], crops[1][12], crops[2][12], crops[3][12], None, None, crops[6][13], crops[7][13], 
         crops[8][13], crops[9][13], crops[10][13], None, crops[12][12], crops[13][12], crops[14][12], crops[15][12]],

        [crops[0][13], crops[1][13], crops[2][13], crops[3][13], crops[4][13], None, crops[6][13], crops[7][13], 
         crops[8][12], crops[9][12], crops[10][12], None, None, None, None, None],

        [crops[0][14], crops[1][14], crops[2][14], crops[3][14], None, None, crops[6][15], crops[7][15], 
         crops[8][15], crops[9][15], None, crops[11][14], crops[12][14], None, crops[14][14], crops[15][14]],

        [crops[0][15], crops[1][15], crops[2][15], crops[3][15], crops[4][15], crops[5][15], crops[6][14], crops[7][14], 
         crops[8][14], crops[9][14], crops[10][14], None, None, None, crops[14][15], crops[15][15]],

        [None, None, None, None, None, None, None, None, 
         crops[8][4], crops[9][4], crops[10][4], crops[11][4], crops[12][4], crops[13][4], crops[14][4], crops[15][4]]

    ]

# Takes an aps filepath and returns a list of 17 vectors containing a set of
# relevant image crops for each body zone
def vectorize_img(image_path):
    img_set = aps_splitter(image_path) 
    groups = group_by_zones(img_set)
    for i in range(17):
        group, img_arr = groups[i], []
        for item in group: 
            if item is None: np.append(img_arr, [-1])
            else: img_arr = np.append(img_arr, item)
        groups[i] = img_arr
    return groups

# Takes path of file containing aps files and path of labels file as input and
# returns a dict containing lists split up by body zones. Each list contains
# pairs of relevant image crops and the corresponding training label
def vectorize(data_path, label_path):

    labels = {}
    with open(label_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            img = row[0].split('_')[0]
            zone = int(row[0].split('_')[1].split('Zone')[1]) - 1
            label = int(row[1])
            if img in labels: labels[img][zone] = label
            else: 
                labels[img] = {}
                labels[img][zone] = label

    body_zone_vectors = {i:[] for i in range(17)}
    for path in os.listdir(data_path):
        if not path.startswith('._') and path.endswith('.aps'):
            img = path.split('.')[0]
            zones = vectorize_img('%s/%s' % (data_path, path))
            if img in labels:
                for i in range(len(zones)):
                    label = labels[img][i]
                    vector = [zones[i], label]
                    body_zone_vectors[i].append(vector)

    xs, ys = {}, {}
    for body_zone in range(17):
        xs[body_zone+1] = [body_zone_vectors[body_zone][i][0] for i in range(len(body_zone_vectors[body_zone]))]
        ys[body_zone+1] = [body_zone_vectors[body_zone][i][1] for i in range(len(body_zone_vectors[body_zone]))]

    return xs, ys
