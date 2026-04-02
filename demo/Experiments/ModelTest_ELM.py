import pandas
import numpy
from Model.ELMModel import ELMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import csv
import matplotlib.pyplot

# Experiment Settings
RANDSEED = 1025 # Random Seed (For Reproduction)
#TRAIN_SIZE = 30 # Single-size Test
TRAIN_SIZE = [5, 10, 15, 20, 25, 30]   # Multi-size Test
RunNum = 30

# Experiment Settings Print
print('====== Experiment Settings ======')
print(f'Random Seed: {RANDSEED}')
print(f'Train Set Size: {TRAIN_SIZE}')
print(f'Number of Runs: {RunNum}')

ExperimentConfigOutput  =  "Results/ExperimentConfig(ModelTest_ELM).csv"
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

        DataTrain, DataTest = train_test_split(Dataset_MLCC, train_size=TrainingSetSize, random_state=RANDSEED+Run)
        #print(DataTrain.shape)
        #print(DataTest.shape)

        #print(DataTrain[Features])
        #print(DataTrain[PredPara])
        #print(DataTrain[Features].shape)
        #print(DataTrain[PredPara].shape)
        scaler = StandardScaler()
        XTrain = scaler.fit_transform(DataTrain[Features].values)
        XTest = scaler.transform(DataTest[Features].values)

        # Regressoion Model
        RegModel = ELMRegressor(RandomState=RANDSEED+Run)

        # Model Fitting
        RegModel.fit(XTrain,DataTrain[PredPara].values)

        # Prediction
        OrigYTrainPred = RegModel.predict(XTrain)
        OrigYTestPred = RegModel.predict(XTest)

        # Evaluation
        OrigTrainRMSE = numpy.sqrt(mean_squared_error(DataTrain[PredPara].values, OrigYTrainPred))
        OrigTrainMAPE = mean_absolute_percentage_error(DataTrain[PredPara].values, OrigYTrainPred)
        OrigTestRMSE = numpy.sqrt(mean_squared_error(DataTest[PredPara].values, OrigYTestPred))
        OrigTestMAPE = mean_absolute_percentage_error(DataTest[PredPara].values, OrigYTestPred)

        OrigTrainRMSEs.append(OrigTrainRMSE)
        OrigTrainMAPEs.append(OrigTrainMAPE)
        OrigTestRMSEs.append(OrigTestRMSE)
        OrigTestMAPEs.append(OrigTestMAPE)

    OrigTrainRMSEs_Mean.append(numpy.mean(OrigTrainRMSEs))
    OrigTrainMAPEs_Mean.append(numpy.mean(OrigTrainMAPEs))
    OrigTestRMSEs_Mean.append(numpy.mean(OrigTestRMSEs))
    OrigTestMAPEs_Mean.append(numpy.mean(OrigTestMAPEs))


StatisticResultsOutput  =  "Results/StatisticResults(ModelTest_ELM).csv"
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