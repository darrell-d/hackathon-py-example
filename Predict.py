import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from UNet_Model import UNET
from Dataset import dataset
from Utils_Predict import *
from Utils_Train import predict_visualization
from Data_Preprocessing import *

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
  PATH = os.path.join(f'/service/data/input/{gate}_model_{0}.pt')
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

  if not os.path.exists(f"/service/data/output/1/figures/Figure_{gate}"):
    os.mkdir(f"/service/data/output/1/figures/Figure_{gate}")
  if not os.path.exists(f"/service/data/output/1/figures/Figure_{gate}/Train_Val"):
    os.mkdir(f"/service/data/output/1/figures/Figure_{gate}/Train_Val")

  for ind in range(path_val.shape[0]):

      data_df_pred, subj_path = mask_to_gate(y_val_list, val_list, x_list, subj_list, x_axis, y_axis, gate, gate_pre, path_raw, worker = 0, idx = ind, seq = seq)

