# purpose: top level script to evaluate an existing model on the test data and compute accuracy.

import os, sys

import torch
# import torch_directml

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
train = load_source('reasonE.train', 'reasonE.train.py')
modelEval = load_source('model.eval', 'model.eval_abox_only.py')
   

if __name__ == '__main__':

  dataPath, modelSavePath, modelSaveNamePrefix, embedDim, lossMargin, negSampleSizeRatio = train.getTrainParams()
  modelSaveNamePrefix = 'model2'
  nIters = 1000

  batchSize = 512
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # device = torch_directml.device()
  print('Device:', device)

  print('Loading data...')
  sys.stdout.flush()
  # dataObj = data.RiceDOReasonData(dataPath, negSampleSizeRatio)
  dataObj = data.TreatOReasonData(dataPath, negSampleSizeRatio, modelSavePath, modelSaveNamePrefix)
  entityCount = dataObj.getEntityCount()
  uConceptCount = dataObj.getUConceptCount()
  bConceptCount = dataObj.getBConceptCount()

  model_path = os.path.join(modelSavePath, f"{modelSaveNamePrefix}.reasonEModel.{nIters}")

  model = torch.load(model_path, map_location="cpu")
  print(f'Loaded model {model_path}')
  model = model.to(device)
  model.eval()

  print('Evaluation...')
  print(' ','with model after '+str(nIters)+' iters of training')

  evalObj = modelEval.ModelEval(dataObj, entityCount, uConceptCount, bConceptCount, embedDim, batchSize, device, model)
  with torch.no_grad():
    evalObj.evalModel()


