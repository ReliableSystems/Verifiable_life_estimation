from scipy.io import loadmat
from torch.utils.data import Dataset, random_split
import numba
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import numpy as np
import torch.optim as optim
from numba import jit
import matplotlib.pyplot as plt
from utils import cumulative_integrate_no_jit,cumulative_integrate_with_jit
from utils import generate_indices

# Below is the base path for the battery data in a matlab format
base_path_matlab = "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab"
# We have 4 different batteries, their names are given in the array below
battery_name_array = ["RW9","RW10","RW11","RW12"]
# We choose one of the batteries and find the path
chosen_battery_name = battery_name_array[0]
chosen_battery_path = base_path_matlab+"/"+chosen_battery_name+".mat"

## We extract the path for the chosen battery's matlab files using scipy
## The variable below named steps contain the entire data for the chosen battery
annots = loadmat(chosen_battery_path)
annots_=annots['data'][0][0]
steps=annots_[0][0]

# The battery used by nasa dataset (18650 li-ion) has 2.2 Ah capacity, Below is the capacity converted to As(unit conversion appropiate for us)
total_battery_capacity = 2.2*3600
one_cycle_capacity = 2*total_battery_capacity #One cycle capacity for the battery is 2xtotal_battery_capacity

## We iterate through the steps below
voltage_stitched = []
current_stitched = []
relative_time_stitched =[]
non_relative_time_stitched =[]
temperature_stitched= []
partial_cycle_stictched =[]
for i in range(len(steps)):
    step=steps[i] #Extracting data for a particular step
    step_type=step[0][0] # Textual categorization of the profile (reference charge,reference discharge, etc), numpy array
    relative_time_array=step[3][0] #Time elapsed since the beginning of a particular step_type, numpy array
    non_relative_time_array=step[2][0] #Time elapsed since the beginning of the experiment, numpy array
    voltage_array=step[4][0] # Voltage characteristics, numpy array
    current_array=step[5][0] # Current characteristics, numpy array
    temperature_array=step[6][0] # Temperature characteristics, numpy array

    #Below we calculate charge passed at a particular time step.
    charge_passed_per_timestep = cumulative_integrate_with_jit(current_array, relative_time_array) # Using numba's jit functionaly the program can be executed faster and in parrallel
    #charge_array = cumulative_integrate_no_jit(current_array, relative_time_array)# Without jit simple implementation also has been provided

    #We calculate the partial cycles done by the battery below
    cycle_array = charge_passed_per_timestep/one_cycle_capacity ## Partial cycles done by the battery

    # Below is a simple data collection procedure which will be used later in the ML models
    voltage_stitched.extend(voltage_array)
    current_stitched.extend(current_array)
    relative_time_stitched.extend(relative_time_array)
    non_relative_time_stitched.extend(non_relative_time_array)
    temperature_stitched.extend(temperature_array)
    partial_cycle_stictched.extend(cycle_array)


# We add the previous charge to the current charge resulting in the cumulative charge across time
for j in range(len(partial_cycle_stictched)-1):
    partial_cycle_stictched[j+1] += partial_cycle_stictched[j]
##Added a new variable for cumulative charge across time
cumulative_cycles = partial_cycle_stictched
# Creating a new array for remaining cycles which will simply be total cycles - cumulative cycles till now
remaining_cycles = []
for j in range(len(cumulative_cycles)):
    remaining_cycles.append(cumulative_cycles[-1]-cumulative_cycles[j])

## We create a pytorch dataset below which will inherit the properties from pytorch's dataset class
class Remaining_cycles_dataset(Dataset):
    def __init__(self, voltage, current, temperature, remaining_cycle,
        non_relative_time,sequence_length=150):

        ## Below are the storages for different parameters
        self.voltage = voltage #storage for voltage parameter
        self.current = current
        self.temperature = temperature
        self.remaining_cycle = remaining_cycle
        self.non_relative_time = non_relative_time

        ## The below items are user inputs which define sequence length and how split up the entire dataset(num_chunks)
        self.sequence_length = sequence_length#
        self.num_chunks = len(temperature)//sequence_length

    def __len__(self):
        return len(self.voltage) - self.sequence_length - 1
        # return self.num_chunks -1

    def __getitem__(self, idx):
        #idx_ = idx*self.sequence_length
        idx_= idx

        current = torch.tensor(self.current[idx_:idx_+self.sequence_length], dtype=torch.float32).unsqueeze(-1)
        voltage = torch.tensor(self.voltage[idx_:idx_+self.sequence_length], dtype=torch.float32).unsqueeze(-1)
        temperature = torch.tensor(self.temperature[idx_:idx_+self.sequence_length], dtype=torch.float32).unsqueeze(-1)
        remaining_cycle = torch.tensor(self.remaining_cycle[idx_:idx_+self.sequence_length], dtype=torch.float32).unsqueeze(-1)

        non_relative_time_tensor = torch.tensor(self.non_relative_time[idx_:idx_+self.sequence_length], dtype=torch.float32).unsqueeze(-1)
        min_time = non_relative_time_tensor.min()
        max_time = non_relative_time_tensor.max()
        scaled_time  = (non_relative_time_tensor-min_time)/(max_time-min_time)## If 0,1 scaling



        return current, voltage, temperature, scaled_time,remaining_cycle

## Below we have the remaining cycles data set initialized with different parameters.
## It will be used to return the synchronized voltage,current,non_relative_time,temperature,re
dataset = Remaining_cycles_dataset(
    voltage_stitched,
    current_stitched,
    temperature_stitched,
    remaining_cycles,
    non_relative_time_stitched,
    sequence_length=150
)

## We define the train,validation and test ration below
total_dataset_size = len(dataset)

## Below code in case we want to generate indices using random permuations

# train_ratio = 0.5
# val_ratio = 0.25
# test_ratio = 0.25
# train_size = int(train_ratio * total_size)
# val_size = int(val_ratio * total_size)
# test_size = total_size - train_size - val_size
# indices = torch.randperm(total_size)
# train_indices = indices[:train_size]
# val_indices = indices[train_size:(train_size + val_size)]
# test_indices = indices[(train_size + val_size):]


## Below we split the data set in the following controlled way
## All the odd indices are in training set (0.5 as fraction of the dataset)
## All the indices divisible by 4 in the validation set
## All the indices not divisible by 4 in the testing set
## 5 element chunk would look like ::   [Train - Test - Train - Validation - Train]

train_indices, val_indices, test_indices = generate_indices(total_size=total_dataset_size)

train_dataset = torch.utils.data.Subset(dataset, train_indices.tolist())
val_dataset = torch.utils.data.Subset(dataset, val_indices.tolist())
test_dataset = torch.utils.data.Subset(dataset, test_indices.tolist())
