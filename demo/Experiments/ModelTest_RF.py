import pandas
import numpy
import matplotlib.pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv

# Experiment Settings
RANDSEED = 1025 # Random Seed (For Reproduction)
#TRAIN_SIZE = 30 # Single-size Test
TRAIN_SIZE = [5, 10, 15, 20, 25, 30]   # Multi-size Test
RunNum = 30
# Save Experiment Settings
ExperimentConfigOutput  =  "Results/ExperimentConfig(ModelTest_RF).csv"
with open(ExperimentConfigOutput,  mode="w", newline="") as f:
    writer = csv.writer(f)
    # Header
    writer.writerow([
        "Parameter",
        "Value"
    ])
    # Data
    writer.writerow(["RandomSeed", RANDSEED])
    writer.writerow(["TrainSize", TRAIN_SIZE])
    writer.writerow(["RunNum", RunNum])

# Dataset
Dataset_MLCC = pandas.read_csv("Dataset/RealDataset.csv", decimal='.', index_col=0)
#print(Dataset_MLCC)
Features = ['PSD-10','Mois','DF*10-4','TC-min','TC-max','TC peak','4SA','PSD-90','PSD-50','Sinter temp','K','D-50']
PredPara = 'RK'

# Statistics Container
OrigTrainRMSEs = [] # Train RMSEs (vary in Training Set Size)
OrigTrainMAPEs = [] # Train MAPEs (vary in Training Set Size)
OrigTestRMSEs = []  # Test RMSEs (vary in Training Set Size)
OrigTestMAPEs = []  # Test MAPEs (vary in Training Set Size

for TrainingSetSize in TRAIN_SIZE:

    # Evaluation Mean Value Container
    OrigTrainRMSE = 0
    OrigTrainMAPE = 0
    OrigTestRMSE = 0
    OrigTestMAPE = 0

    for Run in range(RunNum):

        DataTrain, DataTest = train_test_split(Dataset_MLCC, train_size=TrainingSetSize, random_state=RANDSEED+Run)
        #print(DataTrain.shape)
        #print(DataTest.shape)

        #print(DataTrain[Features])
        #print(DataTrain[PredPara])
        #print(DataTrain[Features].shape)
        #print(DataTrain[PredPara].shape)
        scaler = StandardScaler()
        XTrain = scaler.fit_transform(DataTrain[Features])
        XTest  = scaler.transform(DataTest[Features])

        # Regressoion Model
        RegModel = RandomForestRegressor()

        # Model Fitting
        RegModel.fit(XTrain,DataTrain[PredPara])

        # Prediction
        OrigYTrainPred = RegModel.predict(XTrain)
        OrigYTestPred = RegModel.predict(XTest)

        # Evaluation
        OrigTrainRMSE += numpy.sqrt(mean_squared_error(DataTrain[PredPara], OrigYTrainPred))
        OrigTrainMAPE += mean_absolute_percentage_error(DataTrain[PredPara], OrigYTrainPred)
        OrigTestRMSE += numpy.sqrt(mean_squared_error(DataTest[PredPara], OrigYTestPred))
        OrigTestMAPE += mean_absolute_percentage_error(DataTest[PredPara], OrigYTestPred)

    OrigTrainRMSEMean = OrigTrainRMSE/RunNum
    OrigTrainMAPEMean = OrigTrainMAPE/RunNum
    OrigTestRMSEMean = OrigTestRMSE/RunNum
    OrigTestMAPEMean = OrigTestMAPE/RunNum

    OrigTrainRMSEs.append(OrigTrainRMSEMean)
    OrigTrainMAPEs.append(OrigTrainMAPEMean)
    OrigTestRMSEs.append(OrigTestRMSEMean)
    OrigTestMAPEs.append(OrigTestMAPEMean)

StatisticResultsOutput  =  "Results/StatisticResults(ModelTest_RF).csv"
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
            OrigTrainRMSEs[i],
            OrigTestRMSEs[i],
            OrigTrainMAPEs[i],
            OrigTestMAPEs[i]
        ])

# RMSE Plot
matplotlib.pyplot.plot(TRAIN_SIZE, OrigTestRMSEs, label="Test Data")
matplotlib.pyplot.plot(TRAIN_SIZE, OrigTrainRMSEs, label="Train Data")
matplotlib.pyplot.legend()
matplotlib.pyplot.title("MLCC Dataset Fitting: RMSE (RF)")
matplotlib.pyplot.xlabel("Size of Training Set")
matplotlib.pyplot.ylabel("RMSE Value")
matplotlib.pyplot.savefig("Results/MLCC_RF_RMSE.png")
matplotlib.pyplot.show()
# MAPE Plot
matplotlib.pyplot.plot(TRAIN_SIZE,OrigTestMAPEs, label="Test Data")
matplotlib.pyplot.plot(TRAIN_SIZE,OrigTrainMAPEs, label="Train Data")
matplotlib.pyplot.legend()
matplotlib.pyplot.title("MLCC Dataset Fitting: MAPE (RF)")
matplotlib.pyplot.xlabel("Size of Training Set")
matplotlib.pyplot.ylabel("MAPE Value")
matplotlib.pyplot.savefig("Results/MLCC_RF_MAPE.png")
matplotlib.pyplot.show()