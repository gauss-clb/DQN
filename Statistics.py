import csv
import time
import numpy as np

class Statistics:

    def __init__(self, fileName = 'breakout.csv'):
        print('Write the header of %s: ' % fileName)
        self.csvFile = open(fileName, 'w', newline = '')
        self.csvWriter = csv.writer(self.csvFile)
        self.csvWriter.writerow((
            'epoch',
            'phase',
            'numSteps',
            'numGames',
            'averageReward',
            'minReward',
            'maxReward',
            'averageLoss',
            'lastEpsilon',
            'epochTime'
        ))
        self.csvFile.flush()


    def reset(self):
        self.epochStartTime = time.clock()
        self.numSteps = 0
        self.numGames = 0
        self.gameReward = 0
        self.averageReward = 0.
        self.minReward = 123456789
        self.maxReward = -123456789
        self.averageLoss = 0.
        self.lastEpsilon = 1.

    def statistics(self, action, reward, isTerminal, epsilon):
        self.gameReward += reward
        self.numSteps += 1
        self.lastEpsilon = epsilon
        if isTerminal:
            self.numGames += 1
            self.averageReward += (self.gameReward - self.averageReward) / self.numGames
            self.minReward = min(self.minReward, self.gameReward)
            self.maxReward = max(self.maxReward, self.gameReward)
            self.gameReward = 0


    def lossStatistics(self, loss):
        self.averageLoss += (loss - self.averageLoss) / self.numSteps


    def write(self, epoch, phase):
        currentTime = time.clock()
        epochTime = currentTime - self.epochStartTime

        if self.numGames == 0:
            self.numGames == 1
            self.averageReward = self.gameReward
            self.minReward = self.gameReward
            self.maxReward = self.gameReward
        
        self.csvWriter.writerow((
            epoch,
            phase,
            self.numSteps,
            self.numGames,
            self.averageReward,
            self.minReward,
            self.maxReward,
            self.averageLoss,
            self.lastEpsilon,
            epochTime
        ))
        self.csvFile.flush()
        print('epoch: %d, phase: %s, numSteps: %d, numGames: %d, averageReward: %f, minReward: %d, maxReward: %d, averageLoss: %f, lastEpsilon: %f, epochTime %fs' %
                (epoch, phase, self.numSteps, self.numGames, self.averageReward, self.minReward, self.maxReward, self.averageLoss, self.lastEpsilon, epochTime))

    def close(self):
        self.csvFile.close()