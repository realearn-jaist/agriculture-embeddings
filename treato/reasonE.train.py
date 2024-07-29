# purpose: top level script to train a new model for the specified set of parameters.

import os, sys, math
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
# import data
import data_abox_only as data


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
# modelTrainer = load_source('model.trainer', 'model.trainer.py')
modelTrainer = load_source('model.trainer', 'model.trainer_abox_only.py')


def getTrainParams():
  return dataPath, modelSavePath, modelSaveNamePrefix, embedDim, lossMargin, negSampleSizeRatio

###

dataPath = 'data/'
modelSavePath = 'model/'
modelSaveNamePrefix = 'model2'
embedDim = 100
lossMargin = 1.0
negSampleSizeRatio = 3

if __name__ == '__main__':

  dataPath, modelSavePath, modelSaveNamePrefix, embedDim, lossMargin, negSampleSizeRatio = getTrainParams()

  learningRate = 0.001
  nIters = 1000
  # batchSize = 49336
  batchSize = 512

  print('Training...')
  print(' ','fresh training')
  print(' ',str(nIters)+' iters to do now')

  dataObj = data.TreatOReasonData(dataPath, negSampleSizeRatio)

  trainer = modelTrainer.ModelTrainer(dataObj, dataObj.getEntityCount(), dataObj.getUConceptCount(), dataObj.getBConceptCount(), embedDim)
  logF = open(modelSavePath+'/'+modelSaveNamePrefix+'.'+'log', 'w')
  logF.write('Train: nIters='+str(nIters)+'\n')
  trainer.init(logF)
  loss = trainer.trainIters(batchSize, learningRate, nIters, lossMargin, logF)
  for k, v in loss.items():
    plt.plot(v, label=k)
  plt.legend()
  plt.savefig(os.path.join(modelSavePath, f"{modelSaveNamePrefix}.loss.png"))

  logF.close()

  trainer.saveModel(modelSavePath, modelSaveNamePrefix, str(nIters))
  dataObj.saveEntityConceptMaps(modelSavePath, modelSaveNamePrefix)

  with open(modelSavePath+'/'+modelSaveNamePrefix+'.'+'nIters', 'w') as f:
    f.write("%s\n" % str(nIters))
    f.close()


