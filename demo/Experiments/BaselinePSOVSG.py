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
import seaborn

# Experiment Settings
RANDSEED = 1025 # Random Seed (For Reproduction)
# Dataset Settings
TRAIN_SIZE = [5, 10, 15, 20, 25, 30]   # Multi-size Test
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
TARGET_NVIR = 100 # Target Number of Virtual Sample

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
#print(Dataset_MLCC)

Features = Dataset_MLCC.columns[Dataset_MLCC.columns != 'RK']
PredPara = Dataset_MLCC.columns[Dataset_MLCC.columns == 'RK']
#print("====== Index Verification =======")
#print(Features)
#print(PredPara)

# Statistics Container
OrigTrainRMSEs_Mean = [] # Train RMSEs (vary in Training Set Size)
OrigTrainMAPEs_Mean = [] # Train MAPEs (vary in Training Set Size)
OrigTestRMSEs_Mean = []  # Test RMSEs (vary in Training Set Size)
OrigTestMAPEs_Mean = []  # Test MAPEs (vary in Training Set Size

for TrainingSetSize in TRAIN_SIZE:
    # Evaluation Mean Value Container
    OrigTrainRMSEs = []
    OrigTrainMAPEs = []
    OrigTestRMSEs = []
    OrigTestMAPEs = []
    for Run in range(RunNum):
        # Dataset Processing
        DataTrain, DataTest = train_test_split(Dataset_MLCC, train_size=TrainingSetSize, random_state=RANDSEED+Run)
        #print(DataTrain.shape)
        #print(DataTest.shape)
        #print(DataTrain[Features])
        #print(DataTrain[PredPara])
        #print(DataTrain[Features].shape)
        #print(DataTrain[PredPara].shape)

        # Pre-processing: Normalization
        scaler = StandardScaler()
        DataTrain_Scaled = scaler.fit_transform(DataTrain)
        DataTest_Scaled = scaler.transform(DataTest)
        #print("Shape of DataTrain_Scaled:", DataTrain_Scaled.shape)
        #print(DataTrain_Scaled)
        DataTrain_Scaled_DF = pandas.DataFrame(DataTrain_Scaled,columns=DataTrain.columns)
        DataTest_Scaled_DF = pandas.DataFrame(DataTest_Scaled,columns=DataTest.columns)
        #print("Shape of DataTrain_Scaled_DF:", DataTrain_Scaled_DF.shape)
        #print(DataTrain_Scaled_DF)

        # Regressoion Model
        RegModel = ELMRegressor(RandomState=RANDSEED+Run)
        # Model Fitting
        RegModel.fit(DataTrain_Scaled_DF[Features].values,DataTrain[PredPara].values)
        # Model Prediction
        OrigYTrainPred = RegModel.predict(DataTrain_Scaled_DF[Features].values)
        OrigYTestPred = RegModel.predict(DataTest_Scaled_DF[Features].values)

        # Evaluation
        OrigTrainRMSE = numpy.sqrt(mean_squared_error(DataTrain[PredPara].values, OrigYTrainPred))
        OrigTrainMAPE = mean_absolute_percentage_error(DataTrain[PredPara].values, OrigYTrainPred)
        OrigTestRMSE = numpy.sqrt(mean_squared_error(DataTest[PredPara].values, OrigYTestPred))
        OrigTestMAPE = mean_absolute_percentage_error(DataTest[PredPara].values, OrigYTestPred)
        OrigTrainRMSEs.append(OrigTrainRMSE)
        OrigTrainMAPEs.append(OrigTrainMAPE)
        OrigTestRMSEs.append(OrigTestRMSE)
        OrigTestMAPEs.append(OrigTestMAPE)

        # PSO Domain
        CL = numpy.mean(DataTest_Scaled_DF.loc(axis=1)[Features])
        print(f'CL-{Run}:',CL)

    OrigTrainRMSEs_Mean.append(numpy.mean(OrigTrainRMSEs))
    OrigTrainMAPEs_Mean.append(numpy.mean(OrigTrainMAPEs))
    OrigTestRMSEs_Mean.append(numpy.mean(OrigTestRMSEs))
    OrigTestMAPEs_Mean.append(numpy.mean(OrigTestMAPEs))

for i in range(len(TRAIN_SIZE)):
        print(f"Train Size:{TRAIN_SIZE[i]}|OrigTrainRMSE:{OrigTrainRMSEs_Mean[i]}|OrigTestRMSE:{OrigTestRMSEs_Mean[i]}|OrigTrainMAPE:  {OrigTrainMAPEs_Mean[i]}|OrigTestMAPE:{OrigTestMAPEs_Mean[i]}")

StatisticResultsOutput = "Results/StatisticResults(ModelTest_ELM).csv"
with open(StatisticResultsOutput,  mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Training Set Size",
        "Train RMSE(Mean)",
        "Test RMSE(Mean)",
        "Train MAPE(Mean)",
        "Test MAPE(Mean)"
    ])
    for i in range(len(TRAIN_SIZE)):
        writer.writerow([
            TRAIN_SIZE[i],
            OrigTrainRMSEs_Mean[i],
            OrigTestRMSEs_Mean[i],
            OrigTrainMAPEs_Mean[i],
            OrigTestMAPEs_Mean[i]
        ])

# RMSE Plot
matplotlib.pyplot.plot(TRAIN_SIZE, OrigTestRMSEs_Mean, label="Test Data")
matplotlib.pyplot.plot(TRAIN_SIZE, OrigTrainRMSEs_Mean, label="Train Data")
matplotlib.pyplot.legend(loc=1)
matplotlib.pyplot.title("MLCC Dataset Fitting: RMSE (ELM)")
matplotlib.pyplot.xlabel("Size of Training Set")
matplotlib.pyplot.ylabel("RMSE Value")
matplotlib.pyplot.savefig("Results/MLCC_ELM_RMSE.png")
matplotlib.pyplot.show()
# MAPE Plot
matplotlib.pyplot.plot(TRAIN_SIZE, OrigTestMAPEs_Mean, label="Test Data")
matplotlib.pyplot.plot(TRAIN_SIZE, OrigTrainMAPEs_Mean, label="Train Data")
matplotlib.pyplot.legend(loc=1)
matplotlib.pyplot.title("MLCC Dataset Fitting: MAPE (ELM)")
matplotlib.pyplot.xlabel("Size of Training Set")
matplotlib.pyplot.ylabel("MAPE Value")
matplotlib.pyplot.savefig("Results/MLCC_ELM_MAPE.png")
matplotlib.pyplot.show()

# Fitness Function
# Pre-train Model
# PSO Domain Calculation
CL = numpy.mean(DataTrain_Scaled_DF.loc(axis=1)[Features])
N_L = numpy.count_nonzero(DataTrain_Scaled_DF.loc(axis=1)[Features] < CL, axis=0)
N_U = numpy.count_nonzero(DataTrain_Scaled_DF.loc(axis=1)[Features] > CL, axis=0)
s_p = 1
sk_L = N_L / (N_L + N_U + s_p)
sk_U = N_U / (N_L + N_U + s_p)
print(f"sk_L :: {sk_L}")
LB_TIME = CL - 1/sk_U * (CL - numpy.min(DataTrain_Scaled_DF.loc(axis=1)[Features], axis=0))
UB_TIME = CL + 1/sk_L * (numpy.max(DataTrain_Scaled_DF.loc(axis=1)[Features], axis=0) - CL)
print(f"UB_TIME :\n{UB_TIME}")
print(f"LB_TIME :\n{LB_TIME}")

features_idx = 0
for FeaturesName in Features:
    #type(FeaturesName)
    #print(f'{FeaturesName}')
    CL = DataTrain_Scaled_DF[FeaturesName].mean()
    N_L = numpy.count_nonzero(DataTrain_Scaled_DF[FeaturesName] < CL)
    N_U = numpy.count_nonzero(DataTrain_Scaled_DF[FeaturesName] > CL)
    sk_L = N_L / (N_L + N_U + 1)
    sk_U = N_U / (N_L + N_U + 1)
    LB_MaxMin = DataTrain_Scaled_DF[FeaturesName].min()
    UB_MaxMin = DataTrain_Scaled_DF[FeaturesName].max()
    LB_TIME = CL - 1/sk_U  * (CL - DataTrain_Scaled_DF[FeaturesName].min())
    UB_TIME = CL + 1/sk_L * (DataTrain_Scaled_DF[FeaturesName].max() - CL)
    seaborn.stripplot(x=DataTrain_Scaled_DF[FeaturesName],color='b', label='Train Data')
    seaborn.stripplot(x=DataTest_Scaled_DF[FeaturesName],color='g', label='Test Data')
    matplotlib.pyplot.title(f"Features: {FeaturesName}")
    matplotlib.pyplot.scatter(CL, 0, marker='o', c='r', label='CL')
    matplotlib.pyplot.axvline(LB_MaxMin, c='g', label='MaxMin Boundaries')
    matplotlib.pyplot.axvline(UB_MaxMin, c='g')
    matplotlib.pyplot.axvline(LB_TIME, c='r', label='TIME Boundaries')
    matplotlib.pyplot.axvline(UB_TIME, c='r')
    matplotlib.pyplot.legend(loc=1)
    matplotlib.pyplot.savefig(f"Results/BaselinePSOVSGDomainDemo{features_idx}.png")
    matplotlib.pyplot.show()
    features_idx += 1

# while size(NewSamples) < Nvir:
    # Global Best (Position=None,Fitness=inf)
    # Particles Initialization (Position, Velocity)
    # PSO Argument Settings
    # PSO Search
    # New Samples Append

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
