import sys
import random
import math
from statistics import fmean
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
import matplotlib.pyplot as plt

# Neural Network for Procrastination Evaluation
class ProcrastinationNNExperiment:

    _inboxSuffix: str
    _filesPrefix: str
    _ignoreLateAttempts: int
    _passingGrade: int
    _missingPenalty: int
    _inputFeatures = []
    _debugLevel: int
    _debugNoStudents: int

    #  ["start time","end time","start grade","end grade","max grade","number attempts"]
    _inputStartTime = False
    _inputEndTime = False
    _inputStartGrade = False
    _inputEndGrade = False
    _inputMaxGrade = False
    _inputNumberAttempts = False

    # _attemptsData is complex dictionary of dictionaries
    # _attemptsData[studentId][assignment][attempt] = (grade, time)
    _attemptsData= dict()
    # a list of strings with all assignments in lexicographic order
    _allAssignments = []

    # grades[studentId] = grade      
    _gradesData = dict()
    _numberStudents = 0
    _numberPass = 0
    _numberFail = 0

    # NN features
    _inputSize: int
    _features = dict()
    # 10 groups of features (inputs, target)
    _features10f = []

    _r10fTrainAccuracy = []
    _r10fTrainRecall = []
    _r10fTrainPrecision = []

    def __init__(self, inboxSuffix, filesPrefix, ignoreLateAttempts, passingGrade, missingPenalty, inputFeatures, debugLevel, debugNoStudents):
        # configuration
        self._inboxSuffix = inboxSuffix
        self._filesPrefix = filesPrefix
        self._ignoreLateAttempts = ignoreLateAttempts
        self._passingGrade = passingGrade
        self._missingPenalty = missingPenalty
        self._inputFeatures = inputFeatures
        self._debugLevel = debugLevel
        self._debugNoStudents = debugNoStudents
        # identify computed features
        #  ["start time","end time","start grade","end grade","max grade","number attempts"]
        self._inputStartTime = "start time" in self._inputFeatures
        self._inputEndTime = "end time" in self._inputFeatures
        self._inputStartGrade = "start grade" in self._inputFeatures
        self._inputEndGrade = "end grade" in self._inputFeatures
        self._inputMaxGrade = "max grade" in self._inputFeatures
        self._inputNumberAttempts = "number attempts" in self._inputFeatures
        self._processData()
        self._computeFeatures()

    # process the attempts and create the corresponding dictionary
    def _processAttempts(self):
        # process the attempts
        with open('../inbox'+self._inboxSuffix+'/'+self._filesPrefix+'attempts.csv') as f:
            for line in f:
                line = line.strip()
                # ignore header
                if line[0].isalpha():
                    continue
                # extract column values
                student, assignment, attempt, grade, time = line.split(',')
                student, assignment, attempt, grade, time = int(student), str(assignment), int(attempt), int(grade), int(time)
                # init student if needed
                if student not in self._attemptsData:
                    self._attemptsData[student] = dict()
                # init assignment if needed
                if assignment not in self._attemptsData[student]:
                    self._attemptsData[student][assignment] = dict()
                    if not assignment in self._allAssignments:
                        self._allAssignments.append(assignment)
                if time>self._ignoreLateAttempts:
                    time = time/200.0 + 0.5
                    grade = grade / 100.0
                    self._attemptsData[student][assignment][attempt] = ((grade, time))
        self._allAssignments = sorted(self._allAssignments)

        if self._debugLevel>=1:
            print("Process attempts:")
            print(" - Numer of students: "+str(len(self._attemptsData)))
            print(" - Assignments: "+str(self._allAssignments))
        if self._debugLevel>=2:
            print(" - Sample Students Attempts: ")
            i=0
            for student in self._attemptsData:
                i += 1
                if i>self._debugNoStudents:
                    break
                print("   - "+str(i)+": "+str(student)+" = "+str(self._attemptsData[student]))

    # process the grades and construct the final grades dictionary for the students
    def _processGrades(self):
        if self._debugLevel>=1:
            print("Process grades:")

        # process the final grade
        with open('../inbox'+self._inboxSuffix+'/'+self._filesPrefix+'grades.csv') as g:
            for line in g:
                line = line.strip()
                # ignore header
                if line[0].isalpha():
                    continue
                # extract column values
                student, grade = line.split(',')
                student, grade = int(student), int(grade)
                if student not in self._attemptsData:
                    if self._debugLevel>=2:
                        print(" - student not found: "+str(student)+" w grade: "+str(grade))
                    continue
                # the considered passing grade
                if grade >= self._passingGrade:
                    grade = 0.0
                    self._numberPass += 1
                    self._numberStudents += 1
                else: 
                    grade = 1.0
                    self._numberFail += 1
                    self._numberStudents += 1
                self._gradesData[student] = grade
            
        if self._debugLevel>=1:
            print(" - Grades: "+str(len(self._gradesData)))
            print("    - Pass = "+str(self._numberPass))
            print("    - Fail = "+str(self._numberFail))
        if self._debugLevel>=2:
            i=0
            for grade in self._gradesData:
                print(" - Sample Students Grades: ")
                i += 1
                if i>self._debugNoStudents:
                    break
                print("   - "+str(i)+": "+str(grade)+" = "+str(self._gradesData[grade]))

    # process the input data and extract the needed information
    def _processData(self):
        self._processAttempts()
        self._processGrades()

    # computes the needed features based on the input data
    def _computeFeatures(self):
        if self._debugLevel>=1:
            print("Compute features:")    
        # a dictionary of features 
        # student: (startA1,startA2,startA3,endA3)
        self._inputSize=len(self._allAssignments)*len(self._inputFeatures)
        for student in self._attemptsData:
            if student not in self._gradesData:
                # These students withdrew from the course     
                self._gradesData[student] = 0.0 
                self._numberFail += 1
                self._numberStudents += 1
            input = []
            grade=self._gradesData[student]
            for assignment in self._allAssignments:
                #  ["start time","end time","start grade","end grade","max grade","number attempts"]
                attempts=[]
                hasAttempts = assignment in self._attemptsData[student]
                if hasAttempts:
                    attempts = self._attemptsData[student][assignment].keys()
                    hasAttempts = len(attempts)>0
                if hasAttempts:
                    minAttempt = min(attempts)
                    maxAttempt = max(attempts)
                    maxGrade=0
                    for attempt in self._attemptsData[student][assignment]:
                        g=self._attemptsData[student][assignment][attempt][0]
                        if g>maxGrade:
                            maxGrade=g
                    if self._inputStartTime:
                        input.append(self._attemptsData[student][assignment][minAttempt][1])
                    if self._inputEndTime:
                        input.append(self._attemptsData[student][assignment][maxAttempt][1])
                    if self._inputStartGrade:
                        input.append(self._attemptsData[student][assignment][minAttempt][0])
                    if self._inputEndGrade:
                        input.append(self._attemptsData[student][assignment][maxAttempt][0])
                    if self._inputMaxGrade:
                        input.append(maxGrade)
                    if self._inputNumberAttempts:
                        input.append(len(attempts)/25.0)
                else:
                    if self._inputStartTime:
                        input.append(self._missingPenalty)
                    if self._inputEndTime:
                        input.append(self._missingPenalty)
                    if self._inputStartGrade:
                        input.append(0)
                    if self._inputEndGrade:
                        input.append(0)
                    if self._inputMaxGrade:
                        input.append(0)
                    if self._inputNumberAttempts:
                        input.append(0)                    
            self._features[student]  =  (input,grade)
        
        if self._debugLevel>=1:
            print(" - Features size: "+str(len(self._features)))
        if self._debugLevel>=2:
            print(" - Sample Students Features: ")
            i=0
            for student in self._features:
                i += 1
                if i>self._debugNoStudents:
                    break
                print("   - "+"Student "+str(i)+": "+str(student))
                print("     "+"  - Attempts: "+str(self._attemptsData[student]))
                print("     "+"  - Grades: "+str(self._gradesData[student]))
                print("     "+"  - Features: "+str(self._features[student]))
    
    def getInputSize(self):
        return self._inputSize
    
    # divide the features in 10 groups and prepare inputs and targets for NN
    def _tenFold(self):
        if self._debugLevel >=1:
            print("Creates ten sets of students:")           
        maxSize=int((len(self._features))/10)
        maxSizeFirstIndex=len(self._features)-10*maxSize
        for i in range(10):
            self._features10f.append(([],[]))
        for student in self._features:
            i=-1
            while True:
                i = random.randint(0,9)
                maxAllowed = maxSize
                if i<maxSizeFirstIndex:
                    maxAllowed += 1
                if len(self._features10f[i][0])<maxAllowed:
                    break
            #  ([input,...], [label,...]) 
            self._features10f[i][0].append( self._features[student][0] )
            self._features10f[i][1].append( self._features[student][1] )
        if self._debugLevel>=2:
            print(" - Number of students from features: "+str(len(self._features)))
            print(" - Max number of elements per set: "+str(maxSize))
            for i in range(10):
                print(" - Set "+str(i)+" size = "+str(len(self._features10f[i][0])))

    def _computeTrainTest(self,index):
        if self._debugLevel>=1:
            print("Compute the train and test set:")  
        train=([],[])
        test0=[]
        test1=[]
        train0=[]
        train1=[]
        for i in range(10):
            if i==index:
                test0 += self._features10f[i][0]
                test1 += self._features10f[i][1]
            else:
                train0 += self._features10f[i][0]
                train1 += self._features10f[i][1]
        test=(test0,test1)
        train=(train0,train1)
        if self._debugLevel>=2:
            print(" - Train size: "+str(len(train[0])))
            print(" - Test size: "+str(len(test[0])))
        return train,test

    # run an experiment with the train and test data
    def _experiment(self, train, test, epochs):
        # pnn = ProcrastinationNN(architecture)
        trainDataset = tf.data.Dataset.from_tensor_slices(train)
        testDataset = tf.data.Dataset.from_tensor_slices(test)
        model = tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=[self._inputSize]),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
            ])
        model.compile(loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2),
                      optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
                      metrics=[tf.keras.metrics.Recall(name="recall"),tf.keras.metrics.Precision(name="precision"),'binary_accuracy'])
        history = model.fit(train[0],train[1],epochs=epochs,verbose="",validation_data=test)
        trainAccuracy = history.history['binary_accuracy'][-1]
        trainRecall = history.history['recall'][-1]
        trainPrecision = history.history['precision'][-1]
        self._r10fTrainAccuracy.append(trainAccuracy)
        self._r10fTrainRecall.append(trainRecall)
        self._r10fTrainPrecision.append(trainPrecision)
        print("-----------------------")
        print("Train Recall:   ", trainRecall)
        print("Train Precision:", trainPrecision)
        print("Train Accuracy: ", trainAccuracy)

    
    # run a ten fold cross validation experiment and return the results
    def experiment10foldCV(self, epochs):
        self._tenFold()
        for i in range(10):
            train, test = self._computeTrainTest(i)
            self._experiment(train, test, epochs)
    
    def report(self):
        print()
        print("-----------------------")
        print("Number of students: ",self._numberStudents)
        print(" -            Pass: ",self._numberPass)
        print(" -            Fail: ",self._numberFail)
        print("Train accuracies: ",self._r10fTrainAccuracy)
        print("Train precision: ",self._r10fTrainPrecision)
        print("Train recall: ",self._r10fTrainRecall)
        sumTrainAccuracy=0
        sumTrainPrecision=0
        sumTrainRecall=0
        for i in range(10):
            sumTrainAccuracy += self._r10fTrainAccuracy[i]
            sumTrainPrecision += self._r10fTrainPrecision[i]
            sumTrainRecall += self._r10fTrainRecall[i]
        avgTrainAccuracy= sumTrainAccuracy /10.0
        avgTrainPrecision= sumTrainPrecision /10.0
        avgTrainRecall= sumTrainRecall /10.0
        print("Average train accuracies: ",avgTrainAccuracy)
        print("Average train precision: ",avgTrainPrecision)
        print("Average train recall: ",avgTrainRecall)

def main():
    featuresAll=["start time","end time","start grade","end grade","max grade","number attempts"]
    featuresGrade=["max grade"]
    pnne = ProcrastinationNNExperiment(inboxSuffix="03",filesPrefix="p1-",ignoreLateAttempts=-200, passingGrade=79,missingPenalty=-1,inputFeatures=featuresGrade,debugLevel=2,debugNoStudents=3)
    pnne.experiment10foldCV(1000)
    pnne.report()

if __name__ == '__main__':
    main()


