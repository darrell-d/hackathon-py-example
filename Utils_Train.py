import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from Dataset import dataset

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
  print("=> Saving checkpoint")
  torch.save(state, filename)

def load_checkpoint(checkpoing, model):
  print("=> Loading checkpoint")
  model.load_state_dict(checkpoing["state_dict"])

def get_loaders(path_train, path_test, batch_size, train_transform, test_transform, num_workers = 2, pin_memory = True):
  train_ds = dataset(path_train, train_transform)
  train_loader = DataLoader(train_ds, batch_size = batch_size, num_workers = num_workers, pin_memory = pin_memory, shuffle = True)
  test_ds = dataset(path_test, test_transform)
  test_loader = DataLoader(test_ds, batch_size = batch_size, num_workers = num_workers, pin_memory = pin_memory, shuffle = True)
  return train_loader, test_loader

def train_epoch(loader, model, optimizer, loss_fn, device):
  # loop = tqdm(loader)
  loss_total = 0
  for batch_idx, (data, target, subj) in enumerate(loader):
    data = data.type(torch.float32)
    data = data.to(device = device)
    targets = target.float().unsqueeze(1).to(device = device)

    #forward
    # with torch.cuda.amp.autocast():
    predictions = model(data)
    loss = loss_fn(predictions, targets)    
    loss_total += loss.item()     

    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return loss_total/(batch_idx+1)

def check_accuracy(loader, model, device="cpu"):
  num_correct = 0
  num_pixels = 0
  dice_score = 0
  model.eval()

  with torch.no_grad():
    for x,y, subj in loader:
      x = x.type(torch.float32)
      x = x.to(device)
      y = y.type(torch.float32)
      y = y.to(device).unsqueeze(1)
      preds = torch.sigmoid(model(x))
      preds = (preds > 0.5).float()
      num_correct += (preds == y).sum()
      num_pixels += torch.numel(preds)
      dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

  print(
      f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
  )
  print(f"Dice score: {dice_score/len(loader)}")
  model.train()

  return num_correct/num_pixels*100, dice_score/len(loader)

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

def testing_plot(train_accuracy_all, train_dice_all, test_accuracy_all, test_dice_all, loss_all, gate):

  if not os.path.exists(f"./figures/Figure_{gate}"):
    os.mkdir(f"./figures/Figure_{gate}")

  xpoints = np.linspace(1,len(train_accuracy_all[0]), len(train_accuracy_all[0]))


  plt.figure()
  for accuracy_list in train_accuracy_all:
    accuracy_list = [x.cpu().numpy() for x in accuracy_list]
    plt.plot(xpoints, accuracy_list)
  plt.title(f"Training Accuracy During Traning")
  plt.savefig(f'./figures/Figure_{gate}/Train_Accuracy.png')

  for accuracy_list in test_accuracy_all:
    accuracy_list = [x.cpu().numpy() for x in accuracy_list]
    plt.plot(xpoints, accuracy_list)
  plt.title(f"Validation Accuracy During Traning")
  plt.savefig(f'./figures/Figure_{gate}/Test_Accuracy.png')

  plt.figure()
  for dice_score_list in train_dice_all:
    dice_score_list = [float(x.cpu().numpy()) for x in dice_score_list]
    plt.plot(xpoints, dice_score_list)
  plt.title(f"Training Dice Score During Traning")
  plt.savefig(f'./figures/Figure_{gate}/Train_Dice_Score.png')

  plt.figure()
  for dice_score_list in test_dice_all:
    dice_score_list = [float(x.cpu().numpy()) for x in dice_score_list]
    plt.plot(xpoints, dice_score_list)
  plt.title(f"Validation Dice Score During Traning")
  plt.savefig(f'./figures/Figure_{gate}/Test_Dice_Score.png')

  plt.figure()
  for loss_list in loss_all:
    plt.plot(xpoints, loss_list)
  plt.title(f"Training Loss During Traning")
  plt.savefig(f'./figures/Figure_{gate}/Loss.png')