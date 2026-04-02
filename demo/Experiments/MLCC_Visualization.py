# DatasetVisualization.py
import pandas
import numpy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot
import seaborn

# Experiment Settings
RANDSEED = 1025 # Random Seed (For Reproduction)
# Dataset Settings
TRAIN_SIZE = 5 # Single-size Test

# Dataset (MLCC)
Dataset_MLCC = pandas.read_csv("Dataset/RealDataset.csv", decimal='.', index_col=0)
#print(Dataset_MLCC)

Features = Dataset_MLCC.columns[Dataset_MLCC.columns != 'RK']
PredPara = Dataset_MLCC.columns[Dataset_MLCC.columns == 'RK']
#print("====== Index Verification =======")
#print(Features)
#print(PredPara)

# Dataset Processing
DataTrain, DataTest = train_test_split(Dataset_MLCC, train_size=TRAIN_SIZE, random_state=RANDSEED)
#print(DataTrain)
#print(DataTest)

# Dataset Visualization
features_idx = 0
for FeaturesName in Features:
    #type(FeaturesName)
    #print(f'{FeaturesName}')
    CL = DataTrain[FeaturesName].mean()
    N_L = numpy.count_nonzero(DataTrain[FeaturesName] < CL)
    N_U = numpy.count_nonzero(DataTrain[FeaturesName] > CL)
    sk_L = N_L / (N_L + N_U + 1)
    sk_U = N_U / (N_L + N_U + 1)
    LB_MaxMin = DataTrain[FeaturesName].min()
    UB_MaxMin = DataTrain[FeaturesName].max()
    LB_TIME = CL - 1/sk_U  * (CL - DataTrain[FeaturesName].min())
    UB_TIME = CL + 1/sk_L * (DataTrain[FeaturesName].max() - CL)
    seaborn.stripplot(x=DataTrain[FeaturesName],color='b', label='Train Data')
    seaborn.stripplot(x=DataTest[FeaturesName],color='g', label='Test Data')
    matplotlib.pyplot.title(f"Features: {FeaturesName}")
    matplotlib.pyplot.scatter(CL, 0, marker='o', c='r', label='CL')
    matplotlib.pyplot.axvline(LB_MaxMin, c='g', label='MaxMin Boundaries')
    matplotlib.pyplot.axvline(UB_MaxMin, c='g')
    matplotlib.pyplot.axvline(LB_TIME, c='r', label='TIME Boundaries')
    matplotlib.pyplot.axvline(UB_TIME, c='r')
    matplotlib.pyplot.legend(loc=1)
    matplotlib.pyplot.savefig(f"Results/MLCC_DomainDemo{features_idx}.png")
    matplotlib.pyplot.show()
    features_idx += 1