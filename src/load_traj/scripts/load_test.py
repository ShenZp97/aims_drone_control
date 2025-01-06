import os
import numpy as np
import csv

def read_csv(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        data = [row for row in csv_reader]
    return data

current_folder_path = os.path.dirname(os.path.abspath(__file__))
file_path = current_folder_path+'/new_scvx_traj.csv'
data = read_csv(file_path)
data = np.array(data, dtype=float)

data = data.T

x_traj = data[:, :15]
u_traj = data[:, 15:]
states = x_traj[:, :13]
time_steps = x_traj[:, -2]
controls = u_traj[:, :-1]
print(data.shape)