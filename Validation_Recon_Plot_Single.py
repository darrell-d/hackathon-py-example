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
    
    data_table = pd.read_csv(f'./Raw_Data/{subject}')
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

    data_table = pd.read_csv(f'./prediction/{subject}')

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
    plt.savefig(f'/service/data/output/1/figures/Recon_Sequential_{gate2}_{subject}.png')
    plt.close()

def plot_all(gate1, gate2, x_axis2, y_axis2):
    if not os.path.exists(f"/service/data/output/1/figures/Figure_{gate2}/Recon"):
        os.mkdir(f"/service/data/output/1/figures/Figure_{gate2}/Recon")

    path_val = pd.read_csv(f"./Data/Data_{gate2}/Train_Test_Val/subj_list.csv")
    # find path for raw tabular data
    for subject in path_val.Image:
        plot_one(gate1, gate2, x_axis2, y_axis2, subject)
        


if __name__ == '__main__':
    # setting gates
    gate1 = 'gate1_ir'

    gate2 = 'gate2_cd45'
    x_axis2 = 'Ir193Di___193Ir_DNA2' # x axis in plot
    y_axis2 = 'Y89Di___89Y_CD45'

    n_cross = 8

    plot_all(gate1, gate2, x_axis2, y_axis2)