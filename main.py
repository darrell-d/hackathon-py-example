from Data_Preprocessing import *
from Predict import *
from Validation_Recon_Plot_Single import plot_all
import os
import warnings
warnings.filterwarnings("ignore")

# setting gates
gate_pre_list = [None, 'gate1_ir', 'gate2_cd45', 'gate2_cd45']
gate_list = ['gate1_ir', 'gate2_cd45', 'granulocyte', 'lymphocyte']
x_axis_list = ['Ir191Di___191Ir_DNA1', 'Ir193Di___193Ir_DNA2', 'Yb172Di___172Yb_CD66b', 'Yb172Di___172Yb_CD66b']
y_axis_list = ['Event_length', 'Y89Di___89Y_CD45', 'Y89Di___89Y_CD45', 'Y89Di___89Y_CD45']
path2_lastgate_pred_list = ['./Raw_Data/', 
                    '/service/data/output/1/prediction/',
                    '/service/data/output/1/prediction/',
                    '/service/data/output/1/prediction/',]

# hyperparameter
device = 'cpu'
n_worker = 0

# make directory
if not os.path.exists(f'./Data'):
    os.mkdir(f"./Data")
if not os.path.exists(f'/service/data/output/1/figures'):
    os.mkdir(f"/service/data/output/1/figures")
if not os.path.exists(f'/service/data/output/1/prediction'):
    os.mkdir(f"/service/data/output/1/prediction")
    
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
