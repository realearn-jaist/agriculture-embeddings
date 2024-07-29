# purpose: model trainer class - to train a new model or load and retrain an existing model on the training data for a specific number of iterations and store the resultant model.

import os, sys
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# import torch_directml

import model_abox_only as model


class ModelTrainer:
  def __init__(self, dataObj, entityCount, uConceptCount, bConceptCount, embedDim):
    self.dataObj = dataObj
    self.entityCount = entityCount
    self.uConceptCount = uConceptCount
    self.bConceptCount = bConceptCount
    self.embedDim = embedDim

  def init(self, logF, retrainFlag=False, modelPath=None, modelNamePrefix=None, modelNamePostfix=None):
    if retrainFlag==False:
      self.model = model.ReasonEModel(self.entityCount, self.uConceptCount, self.bConceptCount, self.embedDim)
    else:
      self.model = self.loadModel(modelPath, modelNamePrefix, modelNamePostfix)

    # self.device = torch_directml.device()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', self.device)
    sys.stdout.flush()

    self.model = self.model.to(self.device)

  def trainIters(self, batchSize, learningRate, nIters, lossMargin, logF):
    print('Training iters...')
    sys.stdout.flush()

    modelOpt = torch.optim.Adam(self.model.parameters(), lr = learningRate)
    losses:dict[str, list] = {
      "accLoss": [],
      "uE2CMemberAccLoss": [],
      "bE2CMemberAccLoss": [],
      "uE2CDiscMemberAccLoss": [],
      "bE2CDiscMemberAccLoss": [],
      "uniqENormAccLoss": [],
      "uniqUCBasisAlignAccLoss": [],
      "uniqBCBasisAlignAccLoss": [],
      "uniqUCBasisCountAccLoss": [],
      "uniqBCBasisCountAccLoss": [],
    }
    for it in range(0, nIters):
      # self.dataObj.updateRandomNegAboxTripleList()
      self.dataObj.updateRandomTrainIndexList()
      uE2CMemberAccLoss = 0
      bE2CMemberAccLoss = 0
      uE2CDiscMemberAccLoss = 0
      bE2CDiscMemberAccLoss = 0
      uniqENormAccLoss = 0
      uniqUCBasisAlignAccLoss = 0
      uniqBCBasisAlignAccLoss = 0
      uniqUCBasisCountAccLoss = 0
      uniqBCBasisCountAccLoss = 0
      accLoss = 0
      accCount = 0
      for tI in range(self.dataObj.getTrainDataLen(batchSize)):
        modelOpt.zero_grad()
        aUET, aUCT, nAUET, nAUCT, aBHET, aBTET, aBCT, nABHET, nABTET, nABCT, uniqET, uniqUCT, uniqBCT = self.dataObj.getTrainDataTensor(tI, batchSize)
        
        aUET = aUET.to(self.device)
        aUCT = aUCT.to(self.device)
        nAUET = nAUET.to(self.device)
        nAUCT = nAUCT.to(self.device)
        aBHET = aBHET.to(self.device)
        aBTET = aBTET.to(self.device)
        aBCT = aBCT.to(self.device)
        nABHET = nABHET.to(self.device)
        nABTET = nABTET.to(self.device)
        nABCT = nABCT.to(self.device)
        uniqET = uniqET.to(self.device)
        uniqUCT = uniqUCT.to(self.device)
        uniqBCT = uniqBCT.to(self.device)
        
        model_args = (aUET, aUCT, nAUET, nAUCT, aBHET, aBTET, aBCT, nABHET, nABTET, nABCT, uniqET, uniqUCT, uniqBCT, lossMargin, self.device)
        loss_term = self.model(*model_args)
        uE2CMemberL, bE2CMemberL, uE2CDiscMemberL, bE2CDiscMemberL, uniqENormL, uniqUCBasisAlignL, uniqBCBasisAlignL, uniqUCBasisCountL, uniqBCBasisCountL = loss_term

        uE2CMemberLoss = torch.sum(uE2CMemberL) / len(aUET)
        bE2CMemberLoss = torch.sum(bE2CMemberL)/len(aBHET)
        uE2CDiscMemberLoss = torch.sum(uE2CDiscMemberL)/len(nAUET)
        bE2CDiscMemberLoss = torch.sum(bE2CDiscMemberL)/len(nABHET)
        uniqENormLoss = torch.sum(uniqENormL)/len(uniqET)
        uniqUCBasisAlignLoss = torch.sum(uniqUCBasisAlignL)/len(uniqUCT)
        uniqBCBasisAlignLoss = torch.sum(uniqBCBasisAlignL)/len(uniqBCT)
        uniqUCBasisCountLoss = torch.sum(uniqUCBasisCountL)/len(uniqUCT)
        uniqBCBasisCountLoss = torch.sum(uniqBCBasisCountL)/len(uniqBCT)

        # loss = uE2CMemberLoss + bE2CMemberLoss + uE2CDiscMemberLoss + bE2CDiscMemberLoss + uniqENormLoss + uniqBCBasisAlignLoss + uniqBCBasisCountLoss + uniqUCBasisAlignLoss + uniqUCBasisCountLoss
        loss = uE2CMemberLoss + bE2CMemberLoss + uE2CDiscMemberLoss \
                + bE2CDiscMemberLoss + uniqENormLoss \
                + uniqUCBasisAlignLoss + uniqBCBasisAlignLoss \
                + uniqUCBasisCountLoss + uniqBCBasisCountLoss

        loss.backward()
        modelOpt.step()

        uE2CMemberAccLoss += uE2CMemberLoss.item()
        bE2CMemberAccLoss += bE2CMemberLoss.item()
        uE2CDiscMemberAccLoss += uE2CDiscMemberLoss.item()
        bE2CDiscMemberAccLoss += bE2CDiscMemberLoss.item()
        uniqENormAccLoss += uniqENormLoss.item()
        uniqUCBasisAlignAccLoss += uniqUCBasisAlignLoss.item()
        uniqBCBasisAlignAccLoss += uniqBCBasisAlignLoss.item()
        uniqUCBasisCountAccLoss += uniqUCBasisCountLoss.item()
        uniqBCBasisCountAccLoss += uniqBCBasisCountLoss.item()

        accLoss += loss.item()
        accCount += 1
        # c = accCount

      accLoss /= accCount
      uE2CMemberAccLoss /= accCount
      bE2CMemberAccLoss /= accCount
      uE2CDiscMemberAccLoss /= accCount
      bE2CDiscMemberAccLoss /= accCount
      uniqENormAccLoss /= accCount
      uniqUCBasisAlignAccLoss /= accCount
      uniqBCBasisAlignAccLoss /= accCount
      uniqUCBasisCountAccLoss /= accCount
      uniqBCBasisCountAccLoss /= accCount

      losses["accLoss"].append(accLoss)
      losses["uE2CMemberAccLoss"].append(uE2CMemberAccLoss)
      losses["bE2CMemberAccLoss"].append(bE2CMemberAccLoss)
      losses["uE2CDiscMemberAccLoss"].append(uE2CDiscMemberAccLoss)
      losses["bE2CDiscMemberAccLoss"].append(bE2CDiscMemberAccLoss)
      losses["uniqENormAccLoss"].append(uniqENormAccLoss)
      losses["uniqUCBasisAlignAccLoss"].append(uniqUCBasisAlignAccLoss)
      losses["uniqBCBasisAlignAccLoss"].append(uniqBCBasisAlignAccLoss)
      losses["uniqUCBasisCountAccLoss"].append(uniqUCBasisCountAccLoss)
      losses["uniqBCBasisCountAccLoss"].append(uniqBCBasisCountAccLoss)

      if it % 100 == 0 or it == nIters - 1:
        s = f'iter={str(it)}, {", ".join([f"{k}={v[-1]:.5f}" for k, v in losses.items()])}'
        print(s)
          
    return losses
  
  def loadModel(self, modelPath, modelNamePrefix, modelNamePostfix):
    model = torch.load(modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    print('Loaded model '+modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    return model

  def saveModel(self, modelPath, modelNamePrefix, modelNamePostfix):
    torch.save(self.model, modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)
    print('Saved model '+modelPath+'/'+modelNamePrefix+'.reasonEModel.'+modelNamePostfix)


