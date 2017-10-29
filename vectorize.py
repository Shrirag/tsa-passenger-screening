import matplotlib.pyplot as plt
import numpy as np
import csv
import os

##############################################################################
# Used to turn dataset of 'aps' files and their labels into a set of vectors
# grouped by body zones. Each vector contains all the releveant image crops as
# well as the training label for the body zone
##############################################################################

# Reads headers of 'aps' files
def read_header(infile):

    # Read image header (first 512 bytes)
    h = dict()
    fid = open(infile, 'r+b')
    h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
    h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
    h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
    h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)
    return h

# Reads 'aps' file and returns as numpy array
def read_data(infile):
    
    extension = os.path.splitext(infile)[1]
    
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    fid = open(infile, 'rb')
    fid.seek(512) #skip header
    
    if(h['word_type']==7): #float32
        data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        
    elif(h['word_type']==4): #uint16
        data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        
    data = data * h['data_scale_factor'] #scaling factor
    data = data.reshape(nx, ny, nt, order='F').copy() #make N-d image
    
    return data

# Takes 'aps' file path as input and returns split file from 3D image into 16 2D images
def aps_splitter(path):
    img = read_data(path)
    x, y, z = img.shape
    img_set = np.split(img, z, axis=(len(img.shape)-1))
    img_set = [np.rot90(pic.reshape(x, y)) for pic in img_set]
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
    return [img[160:240, 0:200], img[0:200, 0:160], img[160:240, 330:512], img[0:200, 350:512], img[220:300, 0:512],
            img[300:360, 0:256], img[300:360, 256:512], img[370:450, 0:225], img[370:450, 225:275], img[370:450, 275:512],
            img[450:525, 0:256], img[450:525, 256:512], img[525:600, 0:256], img[525:600, 256:512], img[600:660, 0:256],
            img[600:660, 256:512], img[220:300, 0:512]]

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
   return group_by_zones(img_set)

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
            for i in range(len(zones)):
                label = labels[img][i]
                vector = { 'data': zones[i], 'label': label }
                body_zone_vectors[i].append(vector)

    return body_zone_vectors
