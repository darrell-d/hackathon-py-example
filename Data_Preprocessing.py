import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sn
import scipy
import cv2

def normalize(data, column):
    df_normalize = data[column]
    min = df_normalize.min()
    max = df_normalize.max()
    df_normalize = (df_normalize-min)/(max-min)
    return df_normalize

def matrix_plot(data_df_selected, x_axis, y_axis, pad_number = 100):
    density = np.zeros((101,101))

    data_df_selected = data_df_selected.round(0)
    data_df_selected_count = data_df_selected.groupby([x_axis, y_axis]).size().reset_index(name="count")

    coord = data_df_selected_count[[x_axis, y_axis]]
    coord = coord.to_numpy().round(0).astype(int).T
    coord[0] = 100 - coord[0] # invert position on plot
    coord = list(zip(coord[0], coord[1]))
    replace = data_df_selected_count[['count']].to_numpy()
    for index, value in zip(coord, replace):
        density[index] = value + pad_number # this is to make boundary more recognizable for visualization
    
    index_x = np.linspace(0,100,101).round(2)
    index_y = np.linspace(0,100,101).round(2)
    df_plot = pd.DataFrame(density, index_x, index_y)

    return df_plot

def export_matrix(file_name, x_axis, y_axis, gate_pre, gate, seq = False, raw_path = "./Raw_Data/"):

    if seq:
        data_df = pd.read_csv(os.path.join(f'/service/data/output/1/prediction/', file_name))
        data_df = data_df[data_df[gate_pre + '_pred']==1]
    else:
        data_df = pd.read_csv(os.path.join(raw_path, file_name))

    # # eliminate less than 0
    # data_df = data_df[data_df[x_axis] > 0]  
    # data_df = data_df[data_df[y_axis] > 0]  
    data_df_selected = data_df[[x_axis, y_axis, gate]]
    # if x_axis != "Event_length":
    data_df_selected[x_axis] = normalize(data_df_selected, x_axis)
    data_df_selected[x_axis] = data_df_selected[x_axis]*100
    # if y_axis != "Event_length":
    data_df_selected[y_axis] = normalize(data_df_selected, y_axis)
    data_df_selected[y_axis] = data_df_selected[y_axis]*100

    fig = plt.figure()
    df_plot = matrix_plot(data_df_selected, x_axis, y_axis, 0)
    sn.heatmap(df_plot, vmax = df_plot.max().max()/2, vmin = df_plot.min().min()/2)
    plt.savefig(os.path.join(f'./Data/Data_{gate}/Raw_PNG/', file_name+'.png'))
    np.save(os.path.join(f'./Data/Data_{gate}/Raw_Numpy/', file_name+'.npy'), df_plot)
    plt.close()
    
    fig = plt.figure()
    data_df_masked_2 = data_df_selected[data_df_selected[gate]==1]
    df_plot = matrix_plot(data_df_masked_2, x_axis, y_axis, 0)
    df_plot = df_plot.applymap(lambda x: 1 if x != 0 else 0)
    # check if there is points in gate
    df_plot = df_plot.to_numpy()
    if np.sum(df_plot) > 3:
        df_plot = fill_hull(df_plot)
    sn.heatmap(df_plot, vmax = df_plot.max().max()/2, vmin = df_plot.min().min()/2)
    plt.savefig(os.path.join(f'./Data/Data_{gate}/Mask_PNG/', file_name+'.png'))
    np.save(os.path.join(f'./Data/Data_{gate}/Mask_Numpy/', file_name+'.npy'), df_plot)
    plt.close()

def process_table(x_axis, y_axis, gate_pre, gate, seq = False):   
    if not os.path.exists(f'./Data/Data_{gate}'):
        os.mkdir(f"./Data/Data_{gate}")
        os.mkdir(f"./Data/Data_{gate}/Mask_Numpy")
        os.mkdir(f"./Data/Data_{gate}/Mask_PNG")
        os.mkdir(f"./Data/Data_{gate}/Raw_Numpy")
        os.mkdir(f"./Data/Data_{gate}/Raw_PNG")

    # assign directory
    directory = "./Raw_Data/"
    # iterate over files in
    # that directory
    # extract baseline timepoint
    name_list = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if 'csv' in filename: # excluding health control
                name_list.append(filename)
    name_list_df = pd.DataFrame(name_list, columns =['subject']) 
    name_list_df['subject_id'] = [x.split('.')[0] for x in name_list_df['subject']] # remove .csv
    
    # process the baseline subject
    count=1
    for filename in name_list:
        export_matrix(filename, x_axis, y_axis, gate_pre, gate, seq)
        # print(count)
        count+=1
    print("process table finished")


def filter(path_list):
  path_ = []
  for path in path_list:
    if 'npy' in path:
      path_.append(path)
  return path_

def prepare_prediction_list(gate):
    if not os.path.exists(f"./Data/Data_{gate}/Train_Test_Val"):
        os.mkdir(f"./Data/Data_{gate}/Train_Test_Val")

    imgs = list(sorted(os.listdir(f"./Data/Data_{gate}/Raw_Numpy")))
    masks = list(sorted(os.listdir(f"./Data/Data_{gate}/Mask_Numpy")))

    imgs_ = [f"./Data/Data_{gate}/Raw_Numpy/"+x for x in imgs]
    masks_ = [f"./Data/Data_{gate}/Mask_Numpy/"+x for x in masks]

    imgs = filter(imgs_)
    masks = filter(masks_)
    path = pd.DataFrame(list(zip(imgs, masks)), columns = ['Image','Mask'])

    path.to_csv(f"./Data/Data_{gate}/Train_Test_Val/subj_list.csv", index=False)


def train_test_val_split(gate, n_val = 3, n_cross = 3):
    if not os.path.exists(f"./Data/Data_{gate}/Train_Test_Val"):
        os.mkdir(f"./Data/Data_{gate}/Train_Test_Val")

    imgs = list(sorted(os.listdir(f"./Data/Data_{gate}/Raw_Numpy")))
    masks = list(sorted(os.listdir(f"./Data/Data_{gate}/Mask_Numpy")))

    imgs_ = [f"./Data/Data_{gate}/Raw_Numpy/"+x for x in imgs]
    masks_ = [f"./Data/Data_{gate}/Mask_Numpy/"+x for x in masks]

    imgs = filter(imgs_)
    masks = filter(masks_)
    path = pd.DataFrame(list(zip(imgs, masks)), columns = ['Image','Mask'])

    num_sample = path.shape[0]
    idx_list = list(range(num_sample))
    random.seed(42)
    random.shuffle(idx_list)

    for i in range(n_cross):
        idx_list_cross = idx_list.copy()

        # get index for testing and validation
        remove_index = i*n_val
        last_group = len(idx_list)%n_val

        # pop that index three times to get n_val sequential subject
        test_idx = []
        for j in range(n_val):
            if remove_index < len(idx_list_cross):
                test_idx.append(idx_list_cross.pop(remove_index))

        path_train = path.iloc[idx_list_cross] 
        path_test = path.iloc[test_idx]

        path_train.to_csv(f"./Data/Data_{gate}/Train_Test_Val/Train_{i}.csv", index=False)
        path_test.to_csv(f"./Data/Data_{gate}/Train_Test_Val/Test_{i}.csv", index=False)
        path_test.to_csv(f"./Data/Data_{gate}/Train_Test_Val/Val_{i}.csv", index=False)
    
    print('train val split finished')

def fill_hull(image):
    """
    Compute the convex hull of the given binary image and
    return a mask of the filled hull.
    
    Adapted from:
    https://gist.github.com/stuarteberg/8982d8e0419bea308326933860ecce30
    """
    points = np.argwhere(image).astype(np.int16)
    hull = scipy.spatial.ConvexHull(points)
    convex_points = []
    for vertex in hull.vertices:
        convex_points.append(points[vertex].astype('int64'))
    convex_points = np.array(convex_points)
    convex_points[:, [1, 0]] = convex_points[:, [0, 1]]

    a, b = image.shape
    black_frame = np.zeros([a,b],dtype=np.uint8)
    cv2.fillPoly(black_frame, pts=[convex_points], color=(255, 255, 255))
    black_frame[black_frame == 255] = 1

    return black_frame

if __name__ == "__main__":

    ######
    # test on processing
    #####
    gate = 'gate1_ir'
    y_axis = 'Ir191Di___191Ir_DNA1' # x axis in plot
    x_axis = 'Event_length'
    gate_pre = None
    # root = '/Users/kylee_cj/Dropbox/UPenn/CBICA/Code/2022_10_3_Cytometry_Gating_Software_gate2/'

    if not os.path.exists(f"./Data_{gate}"):
        os.mkdir(f"./Data_{gate}")
        os.mkdir(f"./Data_{gate}/Mask_Numpy")
        os.mkdir(f"./Data_{gate}/Mask_PNG")
        os.mkdir(f"./Data_{gate}/Raw_Numpy")
        os.mkdir(f"./Data_{gate}/Raw_PNG")
    process_table(x_axis, y_axis, gate_pre, gate)

    n_cross = 3 # number of cross validation
    n_val = 3 # number of validation subject
    if not os.path.exists(f"./Data_{gate}/Train_Test_Val"):
        os.mkdir(f"./Data_{gate}/Train_Test_Val")
    train_test_val_split(gate)


    # #####
    # # test mask -> convex hull -> binary mask
    # #####
    # print('start', os.getcwd())
    # data = np.load('./01.T1_Normalized.csv.npy').astype(np.int16)
    # print(data.shape)
    # plt.figure()
    # plt.imshow(data)
    # data_fill = fill_hull(data)  
    # plt.figure()  
    # plt.imshow(data_fill)
    # plt.show()

    
    
    