import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from Data_Preprocessing import normalize
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
  data_df_pred.to_csv(os.path.join(f'/service/data/output/1/prediction/' + subj_path + '.csv'))

  data_df_masked = data_df_pred[data_df_pred[gate + '_pred']==1]

  df_plot = matrix_plot(data_df_masked, x_axis+'_normalized', y_axis+'_normalized')
  df_plot_rotated = ndimage.rotate(df_plot, 90)
  axs[3].imshow(df_plot_rotated, cmap = 'jet')
  axs[3].invert_xaxis()
  axs[3].set_title("Reconstructed Mask")

  plt.savefig(os.path.join(f"/service/data/output/1/figures/Figure_{gate}/Train_Val/" + subj_path + ".png"))

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

if __name__ == "__main__":

  gate = 'neutrophil'
  gate_pre = 'granulocyte'
  y_axis = 'Nd148Di___148Nd_CD16' # x axis in plot
  x_axis = 'Er166Di___166Er_CD294'

  df = pd.read_csv('./prediction/Pred_Results_granulocyte/994574_Normalized.csv')
  print(df.shape)
  data_df_selected = df[df[gate_pre + '_pred']==1].reset_index(drop=True)
  print(data_df_selected.shape)

  print(df[df[x_axis]>0].shape)