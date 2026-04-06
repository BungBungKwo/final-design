import pandas
import numpy
from Model.ELMModel import ELMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pyswarms
from pyswarms.utils.plotters import (plot_cost_history, plot_contour)
import csv
import matplotlib.pyplot
import seaborn

# Experiment Settings
RANDSEED = 1025 # Random Seed (For Reproduction)
# Dataset Settings
TRAIN_SIZE = [20, 25]   # Multi-size Test
#TRAIN_SIZE = [5, 10, 15, 20, 25, 30]   # Multi-size Test
# Multiple Experiment Settings
RunNum = 1
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
OrigTestMAPEs_Mean = []  # Test MAPEs (vary in Training Set Size)
OrigTrainRMSEs_Mean_PSOVSG = [] # Train RMSEs (vary in Training Set Size)
OrigTrainMAPEs_Mean_PSOVSG = [] # Train MAPEs (vary in Training Set Size)
OrigTestRMSEs_Mean_PSOVSG = []  # Test RMSEs (vary in Training Set Size)
OrigTestMAPEs_Mean_PSOVSG = []

for TrainingSetSize in TRAIN_SIZE:
    # Evaluation Mean Value Container
    OrigTrainRMSEs = []
    OrigTrainMAPEs = []
    OrigTestRMSEs = []
    OrigTestMAPEs = []
    OrigTrainRMSEs_PSOVSG = []
    OrigTrainMAPEs_PSOVSG = []
    OrigTestRMSEs_PSOVSG = []
    OrigTestMAPEs_PSOVSG = []

    print(f"====== Training Set Size: {TrainingSetSize} ======")

    for Run in range(RunNum):
        Dataset_MLCC = Dataset_MLCC.apply(pandas.to_numeric)
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
        DataTrain_Scaled = scaler.fit_transform(DataTrain[Features].values)
        DataTest_Scaled = scaler.transform(DataTest[Features].values)
        #print("Shape of DataTrain_Scaled:", DataTrain_Scaled.shape)
        #print(DataTrain_Scaled)
        DataTrain_Scaled_DF = pandas.DataFrame(DataTrain_Scaled,columns=DataTrain[Features].columns)
        DataTest_Scaled_DF = pandas.DataFrame(DataTest_Scaled,columns=DataTest[Features].columns)
        #print("Shape of DataTrain_Scaled_DF:", DataTrain_Scaled_DF.shape)
        #print(DataTrain_Scaled_DF)
        DataTrain_Scaled_DF_WithRK = DataTrain_Scaled_DF.copy()
        DataTrain_Scaled_DF_WithRK[PredPara] = DataTrain[PredPara].values

        # Regressoion Model
        RegModel = ELMRegressor(HiddenLayerNeuronNum=85,RandomState=RANDSEED+Run)
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
        CL = numpy.mean(DataTrain.loc(axis=1)[Features], axis=0)
        #print("====== CL ======")
        #print(CL)
        N_L = numpy.count_nonzero(DataTrain.loc(axis=1)[Features] < CL, axis=0)
        N_U = numpy.count_nonzero(DataTrain.loc(axis=1)[Features] > CL, axis=0)
        s_p = 1
        sk_L = N_L / (N_L + N_U + s_p)
        sk_U = N_U / (N_L + N_U + s_p)
        LB_TIME = CL - 1/sk_U * (CL - numpy.min(DataTrain.loc(axis=1)[Features], axis=0))
        UB_TIME = CL + 1/sk_L * (numpy.max(DataTrain.loc(axis=1)[Features], axis=0) - CL)
        #print(f"UB_TIME :\n{UB_TIME}")
        #print(f"LB_TIME :\n{LB_TIME}")
        # Scale Boundaries
        mean_X = scaler.mean_
        scale_X = scaler.scale_
        LB_TIME_scaled = (LB_TIME - mean_X) / scale_X
        UB_TIME_scaled = (UB_TIME - mean_X) / scale_X
        # Distribution Check
        #print("Train mean (≈0):", numpy.mean(DataTrain_Scaled, axis=0))
        #print("Train std (≈1):", numpy.std(DataTrain_Scaled, axis=0))
        #print("LB scaled:", LB_TIME_scaled)
        #print("UB scaled:", UB_TIME_scaled)
        # Boundaries Check
        #print("Train min:", numpy.min(DataTrain_Scaled, axis=0))
        #print("Train max:", numpy.max(DataTrain_Scaled, axis=0))
        #print("LB_TIME_scaled:", LB_TIME_scaled)
        #print("UB_TIME_scaled:", UB_TIME_scaled)
        # Visualization Check
        #for i in range(len(Features)):
        #    matplotlib.pyplot.figure()
        #    matplotlib.pyplot.hist(DataTrain_Scaled[:, i], bins=20, alpha=0.5, label="Train")
        #    matplotlib.pyplot.axvline(LB_TIME_scaled[i], color='r', label='LB_TIME')
        #    matplotlib.pyplot.axvline(UB_TIME_scaled[i], color='g', label='UB_TIME')
        #    matplotlib.pyplot.title(f"Feature {Features[i]}")
        #    matplotlib.pyplot.legend()
        #    matplotlib.pyplot.show()

        # PSO Fitness Function
        def PSOFitnessFunc(x):
            y = DataTrain['RK'].values
            fitness = numpy.full(x.shape[0], numpy.inf)
            for i in range(x.shape[0]):
                fitness[i] = numpy.min(100 * numpy.abs((y - RegModel.predict(x[i, :].reshape(1, -1))) / y))
            return fitness

        options = {'c1': C1, 'c2': C2, 'w': W_MAX}
        bounds = (LB_TIME_scaled.values, UB_TIME_scaled.values)

        optimizer = pyswarms.single.GlobalBestPSO(
            n_particles=SwarmSize,
            dimensions=len(Features),
            options=options,
            bounds=bounds
        )
        cost, pos = optimizer.optimize(PSOFitnessFunc, iters=MaxIter,verbose=False)
        #pos_re = pos * scale_X + mean_X
        #print(f"pos_re: {pos_re}")
        
        DataVir = pandas.DataFrame(columns=list(Features) + list(PredPara), dtype=float)
        VirSamp = pandas.DataFrame(pos.reshape(1, -1), columns=Features)
        VirSamp[PredPara] = RegModel.predict(pos.reshape(1, -1))
        #print("====== Virtual Sample ======")
        #print(VirSamp)
        DataVir = pandas.concat([DataVir, VirSamp], ignore_index=True)
        #print("====== DataVir ======")
        #print(DataVir)
        DataTrain_PSOVSG = pandas.concat([DataTrain_Scaled_DF_WithRK, DataVir], ignore_index=True)
        #print("====== DataTrain_PSOVSG ======")
        #print(DataTrain_PSOVSG)

        # Model Fitting
        RegModel.fit(DataTrain_PSOVSG[Features].values, DataTrain_PSOVSG[PredPara].values)
        # Model Prediction
        OrigYTrainPred_PSOVSG = RegModel.predict(DataTrain_Scaled_DF[Features].values)
        OrigYTestPred_PSOVSG = RegModel.predict(DataTest_Scaled_DF[Features].values)
        # Evaluation
        OrigTrainRMSE_PSOVSG = numpy.sqrt(mean_squared_error(DataTrain[PredPara].values, OrigYTrainPred_PSOVSG))
        OrigTrainMAPE_PSOVSG = mean_absolute_percentage_error(DataTrain[PredPara].values, OrigYTrainPred_PSOVSG)
        OrigTestRMSE_PSOVSG = numpy.sqrt(mean_squared_error(DataTest[PredPara].values, OrigYTestPred_PSOVSG))
        OrigTestMAPE_PSOVSG = mean_absolute_percentage_error(DataTest[PredPara].values, OrigYTestPred_PSOVSG)
        OrigTrainRMSEs_PSOVSG.append(OrigTrainRMSE_PSOVSG)
        OrigTrainMAPEs_PSOVSG.append(OrigTrainMAPE_PSOVSG)
        OrigTestRMSEs_PSOVSG.append(OrigTestRMSE_PSOVSG)
        OrigTestMAPEs_PSOVSG.append(OrigTestMAPE_PSOVSG)

    OrigTrainRMSEs_Mean.append(numpy.mean(OrigTrainRMSEs))
    OrigTrainMAPEs_Mean.append(numpy.mean(OrigTrainMAPEs))
    OrigTestRMSEs_Mean.append(numpy.mean(OrigTestRMSEs))
    OrigTestMAPEs_Mean.append(numpy.mean(OrigTestMAPEs))

    OrigTrainRMSEs_Mean_PSOVSG.append(numpy.mean(OrigTrainRMSEs_PSOVSG))
    OrigTrainMAPEs_Mean_PSOVSG.append(numpy.mean(OrigTrainMAPEs_PSOVSG))
    OrigTestRMSEs_Mean_PSOVSG.append(numpy.mean(OrigTestRMSEs_PSOVSG))
    OrigTestMAPEs_Mean_PSOVSG.append(numpy.mean(OrigTestMAPEs_PSOVSG))