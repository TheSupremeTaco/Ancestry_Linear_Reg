import numpy as np
import pandas as pd
import copy
import math
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Steps 1-3
class dataInit:
    def __init__(self):
        # Input file IO for training data
        dfDefault = pd.read_csv("TrainingData_N183_p10.csv", header=0)
        # Making a copy for future use
        self.df = dfDefault.__deepcopy__()
        # Init df encoding qualitative as quantitative
        self.df['Ancestry'] = self.df['Ancestry'].map({'African': 0, 'European': 1, 'EastAsian': 2, 'Oceanian': 3, 'NativeAmerican': 4})
        self.df = self.df.iloc[np.random.permutation(len(self.df))]
        self.N = len(self.df)
        self.P = len(self.df.columns)
        self.K = 5
        self.lambdaTuningParm = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4]
        self.a = 10**-5
        # Setup augmented Design matrix(X) N X (p+1)
        self.X = pd.DataFrame(1,index=range(len(self.df)), columns=list('V'))
        self.X = pd.concat([self.X,self.df.iloc[:,:-1]], axis=1)
        self.listTitles = self.X.columns.values.tolist()
        self.Xmean = self.X[self.listTitles].mean()
        self.Xstd = self.X[self.listTitles].std()
        self.X = (self.X[self.listTitles] -self.Xmean)/self.Xstd
        self.X = self.X.fillna(0).to_numpy()
        # Setup Response Matrix(Y) N X K
        tmpResponseVect = self.df.iloc[:, -1]
        self.Y = tmpResponseVect.map(
            {0: np.array([1, 0, 0, 0, 0]), 1: np.array([0, 1, 0, 0, 0]), 2: np.array([0, 0, 1, 0, 0]),
             3: np.array([0, 0, 0, 1, 0]), 4: np.array([0, 0, 0, 0, 1])})
        self.Y = np.stack(self.Y).astype(None)
        self.Ymean = self.Y.mean()
        self.Y -= self.Ymean
        # Init K-Dim Param matrix(B) (p+1) X K
        self.B = np.zeros([self.P, self.K])

    def getDf(self):
        return(self)

class dataInit2:
    def __init__(self):
        # Input file IO for training data
        dfDefault = pd.read_csv("TestData_N111_p10.csv", header=0)
        # Making a copy for future use
        self.df = dfDefault.__deepcopy__()
        # Init df encoding qualitative as quantitative
        self.df['Ancestry'] = self.df['Ancestry'].map({'African': 0, 'European': 1, 'EastAsian': 2, 'Oceanian': 3, 'NativeAmerican': 4})
        self.df = self.df.iloc[np.random.permutation(len(self.df))]
        self.N = len(self.df)
        self.P = len(self.df.columns)
        self.K = 5
        self.lambdaTuningParm = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4]
        self.a = 10**-5
        # Setup augmented Design matrix(X) N X (p+1)
        self.X = pd.DataFrame(1,index=range(len(self.df)), columns=list('V'))
        self.X = pd.concat([self.X,self.df.iloc[:,:-1]], axis=1)
        self.listTitles = self.X.columns.values.tolist()
        self.Xmean = self.X[self.listTitles].mean()
        self.Xstd = self.X[self.listTitles].std()
        self.X = (self.X[self.listTitles] -self.Xmean)/self.Xstd
        self.X = self.X.fillna(0).to_numpy()
        # Setup Response Matrix(Y) N X K
        tmpResponseVect = self.df.iloc[:, -1]
        self.Y = tmpResponseVect.map(
            {0: np.array([1, 0, 0, 0, 0]), 1: np.array([0, 1, 0, 0, 0]), 2: np.array([0, 0, 1, 0, 0]),
             3: np.array([0, 0, 0, 1, 0]), 4: np.array([0, 0, 0, 0, 1])})
        self.Y = np.stack(self.Y).astype(None)
        self.Ymean = self.Y.mean()
        self.Y -= self.Ymean
        # Init K-Dim Param matrix(B) (p+1) X K
        self.B = np.zeros([self.P, self.K])

    def getDf(self):
        return(self)

# Steps 4-7
class linearFit:
    def __init__(self,paramObj: dataInit,choice):
        self.BHat = []
        if(choice == 0):
            for j in range(len(paramObj.lambdaTuningParm)):
                self.B = paramObj.B
                for i in range(10**5):
                    # Init unormalized class probability matrix(U) N X K
                    self.U = np.exp(np.matmul(paramObj.X,self.B))
                    # Init normalized class probability matrix(N) N X K
                    self.P = np.divide(self.U,np.sum(self.U, axis = 1).reshape(-1,1))
                    # Init ease of vect matrix(Z) P X K
                    self.Z = np.zeros([paramObj.P,paramObj.K])
                    self.Z[0]= self.B[0]
                    # Updating Parameter matrix 𝐁 ∶= 𝐁 + 𝛼[𝐗^𝑇(𝐘 − 𝐏) − 2𝜆(𝐁 − 𝐙)]
                    self.B = self.B + paramObj.a*((np.matmul(np.transpose(paramObj.X),(paramObj.Y-self.P)))-2*paramObj.lambdaTuningParm[j]*(self.B-self.Z))
                # Set last update parameter to B hat
                self.BHat.append(self.B[1:])
            self.BHat = np.transpose(np.stack(self.BHat).astype(None))
            print(self.BHat)
        if (choice ==1):
            self.B = paramObj.B
            self.lambdaTuningParm = 10 ** -4
            for i in range(10 ** 5):
                # Init unormalized class probability matrix(U) N X K
                self.U = np.exp(np.matmul(paramObj.X, self.B))
                # Init normalized class probability matrix(N) N X K
                self.P = np.divide(self.U, np.sum(self.U, axis=1).reshape(-1, 1))
                # Init ease of vect matrix(Z) P X K
                self.Z = np.zeros([paramObj.P, paramObj.K])
                self.Z[0] = self.B[0]
                # Updating Parameter matrix 𝐁 ∶= 𝐁 + 𝛼[𝐗^𝑇(𝐘 − 𝐏) − 2𝜆(𝐁 − 𝐙)]
                self.B = self.B + paramObj.a * (
                        (np.matmul(np.transpose(paramObj.X), (paramObj.Y - self.P))) - 2 * self.lambdaTuningParm * (
                            self.B - self.Z))
            # Set last update parameter to B hat
            self.BHat.append(self.B[1:])

class dataInitCV:
    def __init__(self):
        # Input file IO for training data
        dfDefault = pd.read_csv("TrainingData_N183_p10.csv", header=0)
        # Making a copy for future use
        self.df = dfDefault.__deepcopy__()
        # Init df encoding qualitative as quantitative
        self.df['Ancestry'] = self.df['Ancestry'].map({'African': 0, 'European': 1, 'EastAsian': 2, 'Oceanian': 3, 'NativeAmerican': 4})
        self.df = self.df.iloc[np.random.permutation(len(self.df))]
        numFolds = 5
        self.foldsSplit = []
        self.df = self.df.iloc[np.random.permutation(len(self.df))]
        self.df = self.df.reset_index(drop=True)
        split = [37,74,111,148,183]
        self.lambdaTuningParm = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4]
        self.a = 10 ** -5
        for i in range(numFolds):
            # Shuffles df, splits training and valid set, sets the mean and std for each fold
            tmpList = []
            self.valdSet = self.df.iloc[(split[i]-37):split[i]]
            self.trainSet1 = self.df.truncate(before=0, after=(split[i]-37))
            self.trainSet1 = self.trainSet1[:-1]
            self.trainSet2 = self.df.truncate(before=split[i], after=len(self.df))
            self.trainSet = self.trainSet1.append(self.trainSet2)
            tmpList.append([[self.trainSet,],[self.valdSet,]])

            # Setup fold mean and std
            tmpDF = tmpList[0][0][0]
            tmpDF = tmpDF.reset_index(drop=True)
            X = pd.DataFrame(1, index=range(len(tmpDF)), columns=list('V'))
            X = pd.concat([X, tmpDF.iloc[:, :-1]], axis=1)
            listTitles = X.columns.values.tolist()
            self.kfoldMean = X[listTitles].mean()
            self.kfoldStd = X[listTitles].std()

            tmpResponseVect = tmpDF.iloc[:, -1]
            Y = tmpResponseVect.map(
                {0: np.array([1, 0, 0, 0, 0]), 1: np.array([0, 1, 0, 0, 0]), 2: np.array([0, 0, 1, 0, 0]),
                 3: np.array([0, 0, 0, 1, 0]), 4: np.array([0, 0, 0, 0, 1])})
            Y = np.stack(Y).astype(None)
            self.Ymean = Y.mean()

            for j in range(len(tmpList[0])):
                self.tmpDF = tmpList[0][j][0]
                self.tmpDF = self.tmpDF.reset_index(drop=True)
                self.N = len(self.tmpDF)
                self.P = len(self.tmpDF.columns)
                self.K = 5
                self.bHat = []
                self.lambdaTuningParm = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3,
                                         10 ** 4]
                self.a = 10 ** -5
                # Setup augmented Design matrix(X) N X (p+1)
                self.X = pd.DataFrame(1, index=range(len(self.tmpDF)), columns=list('V'))
                self.X = pd.concat([self.X, self.tmpDF.iloc[:, :-1]], axis=1)
                self.listTitles = self.X.columns.values.tolist()
                self.X = (self.X[self.listTitles] - self.kfoldMean) / self.kfoldStd
                self.X = self.X.fillna(0).to_numpy()
                # Setup Response Matrix(Y) N X K
                tmpResponseVect = self.tmpDF.iloc[:, -1]
                self.Y = tmpResponseVect.map(
                    {0: np.array([1, 0, 0, 0, 0]), 1: np.array([0, 1, 0, 0, 0]), 2: np.array([0, 0, 1, 0, 0]),
                     3: np.array([0, 0, 0, 1, 0]), 4: np.array([0, 0, 0, 0, 1])})
                self.Y = np.stack(self.Y).astype(None)
                self.Y -= self.Ymean
                # Init K-Dim Param matrix(B) (p+1) X K
                self.B = np.zeros([self.P, self.K])
                # Creates an array holding everything need for a single fold
                tmpList[0][j].append([self.kfoldMean,self.kfoldStd,self.N,self.P,self.K,self.X,self.Y,self.B,self.bHat])
            self.foldsSplit.append(tmpList)

    def getDf(self):
        return(self)

class linearFitCV:
    def __init__(self,paramObj: dataInitCV):
        self.BHat = []
        self.CVPlot = [[],[]]
        self.lambdaTuningParm = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3,10 ** 4]
        for h in range(len(self.lambdaTuningParm)):
            self.CCE = []
            self.tmpLambda = self.lambdaTuningParm[h]
            for j in range(len(paramObj.foldsSplit)):
                B = paramObj.foldsSplit[j][0][0][1][7]
                self.B = B
                for i in range(10**5):
                    # Init unormalized class probability matrix(U) N X K
                    self.X = paramObj.foldsSplit[j][0][0][1][5]
                    Y = paramObj.foldsSplit[j][0][0][1][6]
                    P = paramObj.foldsSplit[j][0][0][1][3]
                    K = paramObj.foldsSplit[j][0][0][1][4]
                    self.U = np.exp(np.matmul(self.X,self.B))
                    # Init normalized class probability matrix(N) N X K
                    self.P = np.divide(self.U,np.sum(self.U, axis = 1).reshape(-1,1))
                    # Init ease of vect matrix(Z) P X K
                    self.Z = np.zeros([P,K])
                    self.Z[0]= self.B[0]
                    # Updating Parameter matrix 𝐁 ∶= 𝐁 + 𝛼[𝐗^𝑇(𝐘 − 𝐏) − 2𝜆(𝐁 − 𝐙)]
                    self.B = self.B + paramObj.a*((np.matmul(np.transpose(self.X),(Y-self.P)))-2*(self.tmpLambda)*(self.B-self.Z))
                # Set last update parameter to B hat
                paramObj.foldsSplit[j][0][0][1][-1] = np.stack(self.B).astype(None)
                bHat = paramObj.foldsSplit[j][0][0][1][-1]
                # Each fold of CV
                self.X = paramObj.foldsSplit[j][0][1][1][5]
                self.U = np.exp(np.matmul(self.X, bHat))
                # Init normalized class probability matrix(N) N X K
                self.P = self.U/np.sum(self.U, axis=1).reshape(-1, 1)
                Y = paramObj.foldsSplit[j][0][1][1][6]
                self.CCE.append((np.sum(np.sum(np.multiply(Y, np.log10(self.P)), axis=1)))/-37)
            self.CVPlot[0].append(math.log10(self.tmpLambda))
            self.CVPlot[1].append(np.sum(self.CCE) / len(self.CCE))
            print(self.CVPlot)
        plt.plot(self.CVPlot[0],self.CVPlot[1])
        plt.title("Deliverable 2")
        plt.xlabel('Lambda')
        plt.ylabel('CSV Value')
        plt.show()

class predict:
    def __init__(self,paramObj: dataInit,bHat):
        self.bHat = bHat[0]
        self.X = paramObj.X[:,1:]
        K = 5
        N = 111
        #pt1 = np.exp(self.bHat[:,:0]+np.sum(np.matmul(self.X,self.bHat)))
        #pt2 = np.sum(np.exp())
        # Init unormalized class probability matrix(U) N X K
        self.U = np.exp(np.dot(self.X, self.bHat))
        # Init normalized class probability matrix(N) N X K
        self.P = np.divide(self.U,np.sum(self.U, axis=1).reshape(-1,1))
        self.probArr = []
        for i in range(len(self.P)):
            self.probArr.append([np.argmax(self.P[i])+1,round(self.P[i][0]*100),round(self.P[i][1]*100),round(self.P[i][2]*100),round(self.P[i][3]*100),round(self.P[i][4]*100)])
            print(self.probArr[i])


class plotting:
    def __init__(self, Bhat: np.ndarray):
        self.lambdaTuningParm = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4]
        self.objList = Bhat
        for i in range(len(self.lambdaTuningParm)):
            self.lambdaTuningParm[i] = math.log10(self.lambdaTuningParm[i])
        listAncs = ['African', 'European', 'EastAsian', 'Oceanian', 'NativeAmerican']
        figure, axis = plt.subplots(1, 5)
        for i in range(len(listAncs)):
            axis[i].set_title(listAncs[i])
            axis[i].set_xlabel('Lambda')
            axis[i].set_ylabel('B Value')
            for j in range(len(self.objList[i])):
                axis[i].plot(self.lambdaTuningParm,self.objList[i][j])
        plt.show()


# Deliverable one
testObj = dataInit().getDf()
testObj2 = linearFit(testObj,0)
testObj3 = plotting(testObj2.BHat)

# Deliverable two
testObj = dataInitCV().getDf()
testObj2 = linearFitCV(testObj)

# Deliverable three
# lambda value of 10*-4 generated the smallest CV error

# Deliverable four
print("Deliverable four")
testObj = dataInit().getDf()
testObj2 = linearFit(testObj,1)
testObj = dataInit2().getDf()
testObj3 = predict(testObj,testObj2.BHat)

# Deliverable five
# Based on my results data classed as Mexican ancestry was split between native American
# and European percentage and African American were mostly African percentage. Mexicans from history were a mix of
# spaniard explores from Europe and the indigenous people. African Americans are mostly African due to their roots
# originating from importation of Africans in the early colonial periods of America.

# Deliverable six
# Include training and test data and run program