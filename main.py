### dataset
### 2023.11.17
import sys
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class dataset(Dataset):
  def __init__(self, data_dir, transform=None):
    self.data_dir = data_dir
    self.image_dir = data_dir.iloc[:,0].to_list()
    self.mask_dir = data_dir.iloc[:,1].to_list()
    self.transform = transform
  
  def __len__(self):
    return len(self.data_dir)

  def __getitem__(self, index):
    img_path = self.image_dir[index]
    mask_path = self.mask_dir[index]
    image = np.load(img_path,allow_pickle=True).astype('double')
    image = image/image.max()
    mask = np.load(mask_path,allow_pickle=True).astype('double')
    # mask[mask==255] = 1.0

    if self.transform is not None:
      augmentations = self.transform(image=image, mask=mask)
      image = augmentations["image"]
      mask = augmentations["mask"]
      
    return image, mask, img_path

### unet
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf


class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DoubleConv, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False), # kernel, stride, padding, set bias to false because use batchnorm
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False), # kernel, stride, padding, set bias to false because use batchnorm
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
  def forward(self, x):
    return self.conv(x)

class UNET(nn.Module):
  def __init__(
    self, in_channels = 1, out_channels = 1, features = [64,128,256,512],
  ):
    super(UNET, self).__init__()
    self.ups = nn.ModuleList()
    self.downs = nn.ModuleList()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # Down part of UNET
    for feature in features:
      self.downs.append(DoubleConv(in_channels, feature))
      in_channels = feature

    # Up part of UNET
    for feature in reversed(features):
      self.ups.append(
          nn.ConvTranspose2d(
              feature*2, feature, kernel_size = 2, stride = 2,
          )
      )
      self.ups.append(DoubleConv(feature*2, feature))

    self.bottleneck = DoubleConv(features[-1], features[-1]*2)
    self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size = 1)
 
  def forward(self, x):
    skip_connections = []
    for down in self.downs:
      x = down(x)
      skip_connections.append(x)
      x = self.pool(x)
    x = self.bottleneck(x)
    skip_connections = skip_connections[::-1]
    for idx in range(0, len(self.ups), 2):
      x = self.ups[idx](x)
      skip_connection = skip_connections[idx//2]

      # check if the size matches
      if x.shape != skip_connection.shape:
        x = tf.resize(x, size=skip_connection.shape[2:])

      concat_skip = torch.cat((skip_connection, x), dim = 1)
      x = self.ups[idx+1](concat_skip)

    return self.final_conv(x)


### train util
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

def predict_visualization(loader, model, device="cpu"):
  model.eval()
  preds_list = []
  y_list = []
  x_list = []
  subj_list = []
  for idx, (x,y,subj) in enumerate(loader):
    x = x.type(torch.float32)
    x = x.to(device=device)
    with torch.no_grad():
      preds = torch.sigmoid(model(x))
      preds = (preds > 0.5).float()
    preds_list.append(preds)
    y_list.append(y.unsqueeze(1))
    x_list.append(x.unsqueeze(1))
    subj_list.append(subj)

  return preds_list, y_list, x_list, subj_list

### pred util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

from scipy import ndimage
import cv2

def get_pred_label(data_df, x_axis, y_axis, mask, gate, gate_pre=None, seq=False):
    # data_df_selected = data_df[[x_axis, y_axis]]

    if seq:
      # prevent reset index fail
      if 'level_0' in data_df.columns:
         data_df = data_df.drop('level_0', axis=1)

      # keep in-gate prediction data for interpolation
      data_df_selected = data_df[data_df[gate_pre + '_pred']==1].reset_index(drop=True)
      # keep out-gate prediction data for later concat
      data_df_outgate = data_df[data_df[gate_pre + '_pred']==0].reset_index(drop=True)
      data_df_outgate[x_axis+'_normalized'] = 0
      data_df_outgate[y_axis+'_normalized'] = 0
      data_df_outgate[gate + '_pred'] = 0
    else:
      data_df_selected = data_df
    
    # if x_axis != 'Event_length': 
    data_df_selected[x_axis+'_normalized'] = data_df_selected[x_axis].copy()
    data_df_selected[x_axis+'_normalized'] = normalize(data_df_selected, x_axis)*100
    data_df_selected[x_axis+'_normalized'] = data_df_selected[x_axis+'_normalized'].round(0).astype(int)
    # if y_axis != 'Event_length': 
    data_df_selected[y_axis+'_normalized'] = data_df_selected[y_axis].copy()
    data_df_selected[y_axis+'_normalized'] = normalize(data_df_selected, y_axis)*100
    data_df_selected[y_axis+'_normalized'] = data_df_selected[y_axis+'_normalized'].round(0).astype(int)

    # data_df_selected = pd.concat([data_df_selected, data_df[[gate]]], axis = 1)

    index_x = np.linspace(0,100,101).round(0).astype(int).astype(str)
    index_y = np.linspace(0,100,101).round(0).astype(int).astype(str)
    index_x = index_y[::-1] # invert y axis
    df_plot = pd.DataFrame(mask.reshape(101,101), index_x, index_y)

    gate_pred = gate + '_pred'
    # data_df_selected[gate_pred] = [int(df_plot.loc[str(a), str(b)]) for (a, b) in zip(data_df_selected[x_axis], data_df_selected[y_axis])]
    pred_label_list = []
    for i in range(data_df_selected.shape[0]):
      # print(data_df_selected.columns)
      a = data_df_selected.loc[i, x_axis+'_normalized']
      b = data_df_selected.loc[i, y_axis+'_normalized']

      if a > 100 or b > 100:
        print('larger than 100') 
        # outlier - label as 0 and continue
        pred_label_list.append(0)
        continue
      pred_label = int(df_plot.loc[str(a), str(b)])
      true_label = data_df_selected.loc[i, gate]
      pred_label_list.append(pred_label)
    data_df_selected[gate_pred] = pred_label_list

    if seq: 
       data_df_recovered = pd.concat([data_df_selected, data_df_outgate])
    else:
       data_df_recovered = data_df_selected
    return data_df_recovered


def mask_to_gate(y_list, pred_list, x_list, subj_list, x_axis, y_axis, gate, gate_pre, path_raw, worker = 0, idx = 0, seq = False):
  raw_img = x_list[worker][idx]
  mask_img = y_list[worker][idx]
  mask_pred = pred_list[worker][idx]
  subj_path = subj_list[worker][idx]

  # plot
  fig, axs = plt.subplots(1, 4, figsize=(15, 15))

  #plot raw:
  raw_img_rotated = ndimage.rotate(raw_img.cpu().reshape(101,101), 90)
  axs[0].imshow(raw_img_rotated, cmap = 'jet')
  axs[0].invert_xaxis()
  axs[0].axis('off')
  axs[0].set_title("Raw Density Plot")

  #plot true label:
  mask_img_rotated = ndimage.rotate(mask_img.cpu().reshape(101,101), 90)
  axs[1].imshow(mask_img_rotated, cmap = 'jet')
  axs[1].invert_xaxis()
  axs[1].set_title("True Mask")

  #plot prediction:
  mask_pred = mask_pred.cpu().reshape(101,101)
  mask_pred = denoise(mask_pred)
  mask_pred_rotated = ndimage.rotate(mask_pred, 90)
  axs[2].imshow(mask_pred_rotated, cmap = 'jet')
  axs[2].invert_xaxis()
  axs[2].set_title("Predicted Mask")

  # find path for raw tabular data
  substring = os.path.join(f"./Data/Data_{gate}/Raw_Numpy/")
  subj_path = subj_path.split(substring)[1]
  substring = ".csv.npy"
  subj_path = subj_path.split(substring)[0]

  raw_table = pd.read_csv(path_raw + subj_path + '.csv')
  # # remove data point less than 0
  # raw_table = raw_table[raw_table[x_axis] > 0]  
  # raw_table = raw_table[raw_table[y_axis] > 0]  
  raw_table = raw_table.reset_index(drop=True)
  

  data_df_pred = get_pred_label(raw_table, x_axis, y_axis, mask_pred, gate, gate_pre, seq)
  # if '09.T1_Normalized' in subj_path:
  #   print('true')
  data_df_pred.to_csv(os.path.join(f'{sys.argv[2]}/prediction/' + subj_path + '.csv'))

  data_df_masked = data_df_pred[data_df_pred[gate + '_pred']==1]

  df_plot = matrix_plot(data_df_masked, x_axis+'_normalized', y_axis+'_normalized')
  df_plot_rotated = ndimage.rotate(df_plot, 90)
  axs[3].imshow(df_plot_rotated, cmap = 'jet')
  axs[3].invert_xaxis()
  axs[3].set_title("Reconstructed Mask")

  plt.savefig(os.path.join(f"{sys.argv[2]}/figures/Figure_{gate}/Train_Val/" + subj_path + ".png"))

  return data_df_pred, subj_path

def evaluation(data_df_pred, gate):

    accuracy = accuracy_score(data_df_pred[gate], data_df_pred[gate+'_pred'])
    recall = recall_score(data_df_pred[gate], data_df_pred[gate+'_pred'])
    precision = precision_score(data_df_pred[gate], data_df_pred[gate+'_pred'])
    f1 = f1_score(data_df_pred[gate], data_df_pred[gate+'_pred'])

    return accuracy, recall, precision, f1

def matrix_plot(data_df, x_axis, y_axis, pad_number = 100):
    # if x_axis != 'Event_length': 
    #   x_axis = x_axis + '_backup'
    # if y_axis != 'Event_length': 
    #   y_axis = y_axis + '_backup'
    density = np.zeros((101,101))
    data_df_selected = data_df[[x_axis, y_axis]]
    # data_df_selected = data_df_selected.round(2) # round to nearest 0.005
    data_df_selected_count = data_df_selected.groupby([x_axis, y_axis]).size().reset_index(name="count")

    coord = data_df_selected_count[[x_axis, y_axis]]
    # do not normalize event length
    coord = coord.to_numpy().round(0).astype(int).T
    coord[0] = 100 - coord[0] # invert position on plot
    coord = list(zip(coord[0], coord[1]))
    replace = data_df_selected_count[['count']].to_numpy()
    for index, value in zip(coord, replace):
        density[index] = 100 # value + pad_number # this is to make boundary more recognizable for visualization
    
    index_x = np.linspace(0,1,101).round(2)
    index_y = np.linspace(0,1,101).round(2)
    df_plot = pd.DataFrame(density, index_x, index_y)

    return df_plot

def denoise(img):
    img = img.numpy().astype(dtype=np.uint8)
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    if len(contours) <= 1:
       return img
    
    big_contour = max(contours, key=cv2.contourArea)
    little_contour = min(contours, key=cv2.contourArea)


    # get location and area
    area1 = cv2.contourArea(big_contour)
    x1,y1,w1,h1 = cv2.boundingRect(big_contour)
    cv2.rectangle(img, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
    area2 = cv2.contourArea(little_contour)
    x2,y2,w2,h2 = cv2.boundingRect(little_contour)
    cv2.rectangle(img, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)

    plt.imshow(cv2.fillConvexPoly(img, big_contour, color=255))

    img = np.where(img > 1, 1, 0)
    
    return img



### data preprocessing
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

def export_matrix(file_name, x_axis, y_axis, gate_pre, gate, seq = False, raw_path = f"{sys.argv[1]}"):

    if seq:
        data_df = pd.read_csv(os.path.join(f'{sys.argv[2]}/prediction/', file_name))
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
    directory = f"{sys.argv[1]}/"
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

##### predict
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def clean_val_path(subj_path, gate):
    # find path for raw tabular data
    if 'Raw' in subj_path:
      substring = os.path.join(f"./Data/Data_{gate}/Raw_Numpy/")
    else:
      substring = os.path.join(f"./Data/Data_{gate}/Mask_Numpy/")
    subj_path = subj_path.split(substring)[1]
    substring = ".csv.npy"
    subj_path = subj_path.split(substring)[0]
    return subj_path

def unito_gate(x_axis, y_axis, gate, path_raw, num_workers, device, seq = False, gate_pre = None):

  # in sequential predicting, the path_raw is the path for prediction of last gate
  PATH = os.path.join(f'{sys.argv[1]}/{gate}_model_{0}.pt')
  model = UNET().to(device)
  model.load_state_dict(torch.load(PATH))
  model.eval()
  model.to(device)

  test_transforms = A.Compose(
      [
        ToTensorV2(),
      ],
  )

  path_val = pd.read_csv(f"./Data/Data_{gate}/Train_Test_Val/subj_list.csv")

  val_ds = dataset(path_val, test_transforms)
  val_loader = DataLoader(val_ds, batch_size = path_val.shape[0], num_workers = num_workers, pin_memory = True)

  val_list, y_val_list, x_list, subj_list = predict_visualization(val_loader, model, device)

  if not os.path.exists(f"{sys.argv[2]}/figures/Figure_{gate}"):
    os.mkdir(f"{sys.argv[2]}/figures/Figure_{gate}")
  if not os.path.exists(f"{sys.argv[2]}/figures/Figure_{gate}/Train_Val"):
    os.mkdir(f"{sys.argv[2]}/figures/Figure_{gate}/Train_Val")

  for ind in range(path_val.shape[0]):

      data_df_pred, subj_path = mask_to_gate(y_val_list, val_list, x_list, subj_list, x_axis, y_axis, gate, gate_pre, path_raw, worker = 0, idx = ind, seq = seq)



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import datashader as ds
from datashader.mpl_ext import dsshow
from scipy.spatial import ConvexHull

def plot_one(gate1, gate2, x_axis2, y_axis2, subject):

    substring = f"./Data/Data_{gate2}/Raw_Numpy/"   
    subject = subject.split(substring)[1]
    substring = ".npy"
    subject = subject.split(substring)[0]
    
    data_table = pd.read_csv(f'{sys.argv[1]}/{subject}')
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    ###########################################################
    # True
    ###########################################################

    data = data_table.copy()
    if gate1 != None:
        data = data[data[gate1]==1]
    
    in_gate = data[data[gate2] == 1]
    out_gate = data[data[gate2] == 0]
    in_gate = in_gate[[x_axis2, y_axis2]].to_numpy()
    out_gate = out_gate[[x_axis2, y_axis2]].to_numpy()

    if in_gate.shape[0] > 3:
        hull = ConvexHull(in_gate)
        for simplex in hull.simplices:
            ax[0].plot(in_gate[simplex, 0], in_gate[simplex, 1], 'k-')

        density_plot = dsshow(
            data,
            ds.Point(x_axis2, y_axis2),
            ds.count(),
            vmin=0,
            vmax=300,
            norm='linear',
            cmap='jet',
            aspect='auto',
            ax=ax[0],
        )

    # cbar = plt.colorbar(density_plot)
    # cbar.set_label('Number of cells in pixel')
    ax[0].set_xlabel(x_axis2)
    ax[0].set_ylabel(y_axis2)

    if 'Event_length' in x_axis2:
        ax[0].set_xlim([0, 100])
    else:
        ax[0].set_xlim([0, 10])
    if 'Event_length' in y_axis2:
        ax[0].set_ylim([0, 100])
    else:
        ax[0].set_ylim([0, 10])

    ax[0].set_title("Raw Gate 2 Filtered by Gate 1 Plot")


    ###########################################################
    # predict
    ###########################################################
    
    x_axis2_pred = x_axis2
    y_axis2_pred = y_axis2

    if gate1 != None:
        gate1_pred = gate1 + '_pred'
    gate2_pred = gate2 + '_pred'

    data_table = pd.read_csv(f'{sys.argv[2]}/prediction/{subject}')

    data = data_table.copy()
    if gate1 != None:
        data = data[data[gate1_pred]==1]
    in_gate = data[data[gate2_pred] == 1]
    out_gate = data[data[gate2_pred] == 0]
    in_gate = in_gate[[x_axis2_pred, y_axis2_pred]].to_numpy()
    out_gate = out_gate[[x_axis2_pred, y_axis2_pred]].to_numpy()

    if in_gate.shape[0] > 3:
        hull = ConvexHull(in_gate)
        for simplex in hull.simplices:
            ax[1].plot(in_gate[simplex, 0], in_gate[simplex, 1], 'k-')

        density_plot = dsshow(
            data,
            ds.Point(x_axis2_pred, y_axis2_pred),
            ds.count(),
            vmin=0,
            vmax=300,
            norm='linear',
            cmap='jet',
            aspect='auto',
            ax=ax[1],
        )

    # cbar = plt.colorbar(density_plot)
    # cbar.set_label('Number of cells in pixel')
    ax[1].set_xlabel(x_axis2)
    ax[1].set_ylabel(y_axis2)

    if 'Event_length' in x_axis2_pred:
        ax[1].set_xlim([0, 100])
    else:
        ax[1].set_xlim([0, 10])
    if 'Event_length' in y_axis2_pred:
        ax[1].set_ylim([0, 100])
    else:
        ax[1].set_ylim([0, 10])

    ax[1].set_title("Reconstructed Gate 2 Filtered by Gate 1 Plot")

    # save figure
    # plt.savefig(f'./figures/Figure_{gate2}/Recon/Recon_Sequential_{subject}.png')
    plt.savefig(f'{sys.argv[2]}/figures/Figure_{gate2}/Recon/Recon_Sequential_{gate2}_{subject}.png')
    plt.close()

def plot_all(gate1, gate2, x_axis2, y_axis2):
    if not os.path.exists(f"{sys.argv[2]}/figures/Figure_{gate2}/Recon"):
        os.mkdir(f"{sys.argv[2]}/figures/Figure_{gate2}/Recon")

    path_val = pd.read_csv(f"./Data/Data_{gate2}/Train_Test_Val/subj_list.csv")
    # find path for raw tabular data
    for subject in path_val.Image:
        plot_one(gate1, gate2, x_axis2, y_axis2, subject)
        


import os
import warnings
warnings.filterwarnings("ignore")

# setting gates
gate_pre_list = [None, 'gate1_ir', 'gate2_cd45', 'gate2_cd45']
gate_list = ['gate1_ir', 'gate2_cd45', 'granulocyte', 'lymphocyte']
x_axis_list = ['Ir191Di___191Ir_DNA1', 'Ir193Di___193Ir_DNA2', 'Yb172Di___172Yb_CD66b', 'Yb172Di___172Yb_CD66b']
y_axis_list = ['Event_length', 'Y89Di___89Y_CD45', 'Y89Di___89Y_CD45', 'Y89Di___89Y_CD45']
path2_lastgate_pred_list = [f'{sys.argv[1]}/', 
                    f'{sys.argv[2]}/prediction/',
                    f'{sys.argv[2]}/prediction/',
                    f'{sys.argv[2]}/prediction/',]

# hyperparameter
device = 'cpu'
n_worker = 0

# make directory
if not os.path.exists(f'./Data'):
    os.mkdir(f"./Data")
if not os.path.exists(f'{sys.argv[2]}/figures'):
    os.mkdir(f"{sys.argv[2]}/figures")
if not os.path.exists(f'{sys.argv[2]}/prediction'):
    os.mkdir(f"{sys.argv[2]}/prediction")
    
# ###########################################################
# # UNITO
# ###########################################################
for i, (gate_pre, gate, x_axis, y_axis, path_raw) in enumerate(zip(gate_pre_list, gate_list, x_axis_list, y_axis_list, path2_lastgate_pred_list)):
    
    print(f"start UNITO for {gate}")

    # 1. preprocess data
    process_table(x_axis, y_axis, gate_pre, gate, seq = (gate_pre!=None))

    prepare_prediction_list(gate)

    # 3. predict
    unito_gate(x_axis, y_axis, gate, path_raw, n_worker, device, seq = (gate_pre!=None), gate_pre=gate_pre)

    plot_all(gate_pre, gate, x_axis, y_axis)
