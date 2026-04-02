import pandas
import numpy
from Model.ELMModel import ELMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pyswarms
import csv
import matplotlib.pyplot

# Experiment Settings
RANDSEED = 1025 # Random Seed (For Reproduction)
# Dataset Settings
TRAIN_SIZE = 30 # Single-size Test
#TRAIN_SIZE = [5, 10, 15, 20, 25, 30]   # Multi-size Test
# Multiple Experiment Settings
RunNum = 30
# ELM Settings
HIDDEN_LAYER_NEURON_NUM = 85
# PSO Settings
SwarmSize = 50  # Swarm Size
MaxIter = 500   # Maximum Iterations
W_MAX = 0.9     # Inertia Weight (Max)
W_MIN = 0.4     # Inertia Weight (Min)
C1 = 2          # Learning Factor C1
C2 = 2          # Learning Factor C2
FITNESS_DESIRED = 10**-6    # Desired value of Fitness
# VSG Settings
TARGET_NVIR = 15 # Target Number of Virtual Sample

# Experiment Settings Print
print('====== Experiment Settings ======')
print(f'Random Seed: {RANDSEED}')
print(f'Train Size: {TRAIN_SIZE}')
print(f'Number of Runs: {RunNum}')
print(f'HiddenLayerNeuronNumber: {HIDDEN_LAYER_NEURON_NUM}')
print(f'Swarm Size: {SwarmSize}')
print(f'Maximum Iteration (PSO): {MaxIter}')
print(f'Inertia Weight (Max): {W_MAX}')
print(f'Inertia Weight (Min): {W_MIN}')
print(f'Fitness Precision: {FITNESS_DESIRED}')
print(f'Target Number of Virtual Sample: {TARGET_NVIR}')

ExperimentConfigOutput  =  "Results/ExperimentConfig(BaselinePSOVSG).csv"
with open(ExperimentConfigOutput,  mode="w", newline="") as f:
    writer = csv.writer(f)
    # Header
    writer.writerow(["Parameter", "Value"])
    # Data
    writer.writerow(["Random Seed", RANDSEED])
    writer.writerow(["Train Size", TRAIN_SIZE])
    writer.writerow(["Number of Runs", RunNum])
    writer.writerow(["Hidden Layer Neuron  Number", HIDDEN_LAYER_NEURON_NUM])
    writer.writerow(["Swarm Size", SwarmSize])
    writer.writerow(["Maximum Iteration (PSO)", MaxIter])
    writer.writerow(["Inertia Weight (Max)", W_MAX])
    writer.writerow(["Inertia Weight (Min)", W_MIN])
    writer.writerow(["Fitness Precision", FITNESS_DESIRED])
    writer.writerow(["Target Number of Virtual Sample", TARGET_NVIR])

# Dataset (MLCC)
Dataset_MLCC = pandas.read_csv("Dataset/RealDataset.csv", decimal='.', index_col=0)
print(Dataset_MLCC)

Features = Dataset_MLCC.columns[Dataset_MLCC.columns != 'RK']
PredPara = Dataset_MLCC.columns[Dataset_MLCC.columns == 'RK']
print("====== Index Verification =======")
print(Features)
print(PredPara)

# Dataset Processing
DataTrain, DataTest = train_test_split(Dataset_MLCC, train_size=TRAIN_SIZE, random_state=RANDSEED)
print(DataTrain)
print(DataTest)




## Pre-processing: Normalization
#scaler = StandardScaler()
#XTrain = scaler.fit_transform(DataTrain[Features].values)
#XTest = scaler.transform(DataTest[Features].values)
##print(f'Train Shape: {XTrain.shape}')
##print(f'Test Shape: {XTest.shape}')
##print(XTrain)
##print(XTest)
#
## Regressoion Model
#RegModel = ELMRegressor(RandomState=RANDSEED)
#
## Model Fitting
#RegModel.fit(XTrain,DataTrain[PredPara].values)
#
## Prediction
#OrigYTrainPred = RegModel.predict(XTrain)
#OrigYTestPred = RegModel.predict(XTest)
#
## Evaluation
#OrigTrainRMSE = numpy.sqrt(mean_squared_error(DataTrain[PredPara].values, OrigYTrainPred))
#OrigTrainMAPE = mean_absolute_percentage_error(DataTrain[PredPara].values, OrigYTrainPred)
#OrigTestRMSE = numpy.sqrt(mean_squared_error(DataTest[PredPara].values, OrigYTestPred))
#OrigTestMAPE = mean_absolute_percentage_error(DataTest[PredPara].values, OrigYTestPred)
#
#print(f"OrigTrainRMSE: {OrigTrainRMSE} | OrigTrainMAPE: {OrigTrainMAPE}")
#
## PSO Domain
#PSOBound_LB = XTrain.min(axis=0)
#PSOBound_UB = XTrain.max(axis=0)
##print("====== PSO Boundaries ======")
##print(f"PSO Bound(LB): {PSOBound_LB}")
##print(f"PSO Bound(UB): {PSOBound_UB}")
#
## Fitness Function
##def PSOFitnessFunc(x):
#
#
#
##CL = Numpy.mean(data_train.loc(axis=1)['X', 'Y'])
##N_L = np.count_nonzero(data_train.loc(axis=1)['X', 'Y'] <= CL, axis=0)
##N_U = np.count_nonzero(data_train.loc(axis=1)['X', 'Y'] >= CL, axis=0)
##s_p = 1
##sk_L = N_L / (N_L + N_U + s_p)
##sk_U = N_U / (N_L + N_U + s_p)
#