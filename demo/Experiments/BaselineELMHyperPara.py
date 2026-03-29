import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Model.ELMModel import ELMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import csv
import matplotlib.pyplot

# Experiment Settings
RANDSEED = 1025 # Random Seed (For Reproduction)
# Dataset Settings
TRAIN_SIZE = 30 # Single-size Test
# Multiple Experiment Settings
RunNum = 30
# HyperParameter Test: ELM Hidden Layer Neuron Number
HIDDEN_LAYER_NEURON_NUM_MIN = 5
HIDDEN_LAYER_NEURON_NUM_MAX = 100
TEST_STEP = 1
HiddenNeuronNumList = range(HIDDEN_LAYER_NEURON_NUM_MIN, HIDDEN_LAYER_NEURON_NUM_MAX+1, TEST_STEP)

# Experiment Settings Print
print('====== Experiment Settings ======')
print(f'Random Seed: {RANDSEED}')
print(f'Train Size: {TRAIN_SIZE}')
print(f'Number of Runs: {RunNum}')
print(f'Range of Hidden Layer: {HIDDEN_LAYER_NEURON_NUM_MIN}-{HIDDEN_LAYER_NEURON_NUM_MAX} (Setp:{TEST_STEP})')

# Experiment Settings Save
ExperimentConfigOutput  =  "Results/ExperimentConfig(BaselineELMHyperPara).csv"
with open(ExperimentConfigOutput,  mode="w", newline="") as f:
    writer = csv.writer(f)
    # Header
    writer.writerow(["Parameter", "Value"])
    # Data
    writer.writerow(["RandomSeed", RANDSEED])
    writer.writerow(["Training Set Size", TRAIN_SIZE])
    writer.writerow(["Number of run", RunNum])
    writer.writerow(["HiddenLayerNeuronNumberMin", HIDDEN_LAYER_NEURON_NUM_MIN])
    writer.writerow(["HiddenLayerNeuronNumberMax", HIDDEN_LAYER_NEURON_NUM_MAX])
    writer.writerow(["TestStep", TEST_STEP])

# Dataset
Dataset_MLCC = pandas.read_csv("Dataset/RealDataset.csv", decimal='.', index_col=0)
#print(Dataset_MLCC)
Features = ['PSD-10','Mois','DF*10-4','TC-min','TC-max','TC peak','4SA','PSD-90','PSD-50','Sinter temp','K','D-50']
PredPara = 'RK'

# Statistic Container
HiddenLayerNeuronNums = []
OrigTrainMAPEs_Mean = [] # Index：Hidden Layer Num
OrigTestMAPEs_Mean = [] # Index：Hidden Layer Num
# Best Hidden Layer Neuron Number Container
BestHiddenNeuronNum = None
OrigTestMAPE_BestMean = float("inf")

for NeuronNum in HiddenNeuronNumList:
    OrigTrainMAPEs = [] # Index：Run
    OrigTestMAPEs = [] # Index：Run
	
    for Run in range(RunNum):
        # Dataset Processing
        DataTrain, DataTest = train_test_split(Dataset_MLCC, train_size=TRAIN_SIZE, random_state=RANDSEED+Run)
        # Pre-processing: Normalization
        scaler = StandardScaler()
        XTrain = scaler.fit_transform(DataTrain[Features].values)
        XTest = scaler.transform(DataTest[Features].values)
        # Regressoion Model
        RegModel = ELMRegressor(HiddenLayerNeuronNum=NeuronNum,RandomState=RANDSEED+Run)
        # Model Fitting
        RegModel.fit(XTrain,DataTrain[PredPara].values)
        # Prediction
        OrigYTrainPred = RegModel.predict(XTrain)
        OrigYTestPred = RegModel.predict(XTest)
        # Evaluation 
        OrigTrainMAPE = mean_absolute_percentage_error(DataTrain[PredPara].values, OrigYTrainPred)
        OrigTestMAPE = mean_absolute_percentage_error(DataTest[PredPara].values, OrigYTestPred)

        OrigTrainMAPEs.append(OrigTrainMAPE)
        OrigTestMAPEs.append(OrigTestMAPE)

    OrigTrainMAPEs_Mean.append(numpy.mean(OrigTrainMAPEs))
    OrigTestMAPEs_Mean.append(numpy.mean(OrigTestMAPEs))

print("====== Results ======")
for i in range(len(HiddenNeuronNumList)):
    print(f"Hidden Layer Neuron Number:{HiddenNeuronNumList[i]} | TrainMAPE:{OrigTrainMAPEs_Mean[i]} | TestMAPE:{OrigTestMAPEs_Mean[i]}")

StatisticResultsOutput  =  "Results/StatisticResults(BaselineELMHyperPara).csv"
with open(StatisticResultsOutput,  mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Hidden Layer Neuron Number",
        "Train MAPE(Mean)",
        "Test MAPE(Mean)"
    ])
    for i in range(len(HiddenNeuronNumList)):
        writer.writerow([
            HiddenNeuronNumList[i],
            OrigTrainMAPEs_Mean[i],
            OrigTestMAPEs_Mean[i]
    ])

# Hidden Layer Neurons Number vs. Test MAPE
ResultPlotOutput = "Results/BaselineHyperPara.png"
matplotlib.pyplot.figure()
matplotlib.pyplot.plot(HiddenNeuronNumList, OrigTestMAPEs_Mean)
matplotlib.pyplot.xlabel("Hidden Layer Neuron Number")
matplotlib.pyplot.ylabel("Test MAPE")
matplotlib.pyplot.title("Hidden Layer Size vs. Test MAPE")
matplotlib.pyplot.savefig(ResultPlotOutput)
matplotlib.pyplot.show()