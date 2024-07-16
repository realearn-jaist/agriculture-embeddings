# purpose: top level script to retrain an existing model for the specified set of parameters.

import os, sys, math
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

import data


import importlib
def load_source(modname, filename):
    loader = importlib.machinery.SourceFileLoader(modname, filename)
    spec = importlib.util.spec_from_file_location(modname, filename, loader=loader)
    module = importlib.util.module_from_spec(spec)
    # The module is always executed and not cached in sys.modules.
    # Uncomment the following line to cache the module.
    # sys.modules[module.__name__] = module
    loader.exec_module(module)
    return module
train = load_source('reasonE.train', 'reasonE.train.py')
modelTrainer = load_source('model.trainer', 'model.trainer.py')


###

if __name__ == '__main__':

  dataPath, modelSavePath, modelSaveNamePrefix, embedDim, lossMargin, negSampleSizeRatio = train.getTrainParams()

  learningRate = 0.00001
  nIters = 1000
  batchSize = 49336
  #batchSize = 881

  with open(modelSavePath+'/'+modelSaveNamePrefix+'.'+'nIters','r') as f:
    lst = f.readlines()
    oldNIters = int(lst[0].strip())
    totalNIters = oldNIters + nIters

  print('Retraining...')
  print(' ',str(oldNIters)+' iters so far')
  print(' ',str(nIters)+' iters to do now')

  dataObj = data.RiceDOReasonData(dataPath, negSampleSizeRatio, modelSavePath, modelSaveNamePrefix)
  sys.stdout.flush()

  trainer = modelTrainer.ModelTrainer(dataObj, dataObj.getEntityCount(), dataObj.getUConceptCount(), dataObj.getBConceptCount(), embedDim)
  logF = open(modelSavePath+'/'+modelSaveNamePrefix+'.'+'log', 'a')
  logF.write('Retrain: nIters='+str(nIters)+'\n')
  trainer.init(logF, True, modelSavePath, modelSaveNamePrefix, str(oldNIters))
  trainer.trainIters(batchSize, learningRate, nIters, lossMargin, logF)
  sys.stdout.flush()
  logF.close()

  trainer.saveModel(modelSavePath, modelSaveNamePrefix, str(totalNIters))

  with open(modelSavePath+'/'+modelSaveNamePrefix+'.'+'nIters', 'w') as f:
    f.write("%s\n" % str(totalNIters))
    f.close()


