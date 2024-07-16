# purpose: model definition class - to define model and forward step.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReasonEModel(nn.Module):
  def __init__(self, entityCount, uConceptCount, bConceptCount, embedDim):
    super(ReasonEModel, self).__init__()
    self.embedDim = embedDim
    self.entityCount = entityCount
    self.uConceptCount = uConceptCount
    self.bConceptCount = bConceptCount

    self.baseMat = torch.FloatTensor(torch.eye(embedDim))
    self.entityEmbed = nn.Embedding(entityCount, embedDim)
    self.uConceptEmbed = nn.Embedding(uConceptCount, embedDim)
    self.bConceptHEmbed = nn.Embedding(bConceptCount, embedDim)
    self.bConceptTEmbed = nn.Embedding(bConceptCount, embedDim)
    nn.init.xavier_uniform_(self.entityEmbed.weight)
    nn.init.xavier_uniform_(self.bConceptHEmbed.weight)
    nn.init.xavier_uniform_(self.bConceptTEmbed.weight)
    self.entityEmbed.weight.data = F.normalize(self.entityEmbed.weight.data, p=2, dim=1)
    self.uConceptEmbed.weight.data = F.normalize(self.uConceptEmbed.weight.data, p=2, dim=1)
    self.bConceptHEmbed.weight.data = F.normalize(self.bConceptHEmbed.weight.data, p=2, dim=1)
    self.bConceptTEmbed.weight.data = F.normalize(self.bConceptTEmbed.weight.data, p=2, dim=1)

  def forward(self, aUE, aUC, nAUE, nAUC, aBHE, aBTE, aBC, nABHE, nABTE, nABC, uniqE, uniqUC, uniqBC, lossMargin, device):
    aUEE = self.entityEmbed(aUE)
    aUCE = self.uConceptEmbed(aUC)
    nAUEE = self.entityEmbed(nAUE)
    nAUCE = self.uConceptEmbed(nAUC)

    aBHEE = self.entityEmbed(aBHE)
    aBTEE = self.entityEmbed(aBTE)
    aBCHE = self.bConceptHEmbed(aBC)
    aBCTE = self.bConceptTEmbed(aBC)

    nABHEE = self.entityEmbed(nABHE)
    nABTEE = self.entityEmbed(nABTE)
    nABCHE = self.bConceptHEmbed(nABC)
    nABCTE = self.bConceptTEmbed(nABC)

    uniqEE = self.entityEmbed(uniqE)
    uniqUCE = self.uConceptEmbed(uniqUC)
    uniqBCHE = self.bConceptHEmbed(uniqBC)
    uniqBCTE = self.bConceptTEmbed(uniqBC)

    zero = torch.FloatTensor([0.0])
    zero = zero.to(device)
    one = torch.FloatTensor([1.0])
    one = one.to(device)
    # halfDim = Variable(torch.FloatTensor([self.embedDim/2.0]))
    # halfDim = halfDim.to(device)
    margin = torch.FloatTensor([lossMargin])
    margin = margin.to(device)

    # Enforce unary membership, first term in eq. 4
    tmpUE2C = (one-aUCE)*aUEE
    uE2CMemberL = torch.sum(tmpUE2C*tmpUE2C, dim=1)

    # Enforce binary membership, first term in eq. 4
    tmpBE2CH = (one-aBCHE)*aBHEE
    tmpBE2CT = (one-aBCTE)*aBTEE
    bE2CMemberL = torch.sum(tmpBE2CH*tmpBE2CH, dim =1) + torch.sum(tmpBE2CT*tmpBE2CT, dim=1)

    # negative sampling for unary
    tmpNUE2C = (one-nAUCE)*nAUEE
    tmpNUL = torch.sum(tmpNUE2C*tmpNUE2C, dim=1)
    uE2CDiscMemberL = torch.max(margin-tmpNUL, zero)

    # negative sampling for binary
    tmpNBE2CH = (one-nABCHE)*nABHEE
    tmpNBE2CT = (one-nABCTE)*nABTEE
    tmpNBL = torch.sum(tmpNBE2CH* tmpNBE2CH, dim=1) + torch.sum(tmpNBE2CT*tmpNBE2CT, dim=1)
    bE2CDiscMemberL = torch.max(margin-tmpNBL, zero)

    # Enforce unit length, last term in eq. 4 for unary
    tmpE = torch.sum(uniqEE*uniqEE, dim=1) - one
    uniqENormL = tmpE*tmpE

    # Enforce y_i is binary vector. eq. 3 for unary
    tmpUC = uniqUCE*(one-uniqUCE)
    uniqUCBasisAlignL = torch.sum(tmpUC*tmpUC, dim=1)

    # Enforce y_i to not be all zeros for unary
    tmpUCDim = torch.sum(torch.abs(uniqUCE), dim=1)
    uniqUCBasisCountL = torch.max(one-tmpUCDim, zero)
    
    # Ensuring that z_i is binary vector. eq. 3 for binary
    tmpBCH = uniqBCHE*(one-uniqBCHE)
    tmpBCT = uniqBCTE*(one-uniqBCTE)
    uniqBCBasisAlignL = torch.sum(tmpBCH*tmpBCH, dim=1) + torch.sum(tmpBCT*tmpBCT, dim=1)

    # Enforce z_i to not be all zeros. eq. 2 for binary
    tmpBCHDim = torch.sum(torch.abs(uniqBCHE), dim=1)
    tmpBCHL = torch.max(one-tmpBCHDim, zero)
    tmpBCTDim = torch.sum(torch.abs(uniqBCTE), dim=1)
    tmpBCTL = torch.max(one-tmpBCTDim, zero)
    uniqBCBasisCountL = tmpBCHL + tmpBCTL
    
    return uE2CMemberL, bE2CMemberL, uE2CDiscMemberL, bE2CDiscMemberL, uniqENormL, uniqUCBasisAlignL, uniqBCBasisAlignL, uniqUCBasisCountL, uniqBCBasisCountL
    
  def getEntityEmbedding(self, e):
    return self.entityEmbed(e)

  def getUConceptEmbedding(self, c):
    return self.uConceptEmbed(c)

  def getBConceptHEmbedding(self, c):
    return self.bConceptHEmbed(c)

  def getBConceptTEmbedding(self, c):
    return self.bConceptTEmbed(c)

  def getBaseMat(self):
    return self.baseMat


class EvalOnThisCodeModel:
      
  def getEntityEmbedding(self, e):
    raise NotImplementedError()

  def getUConceptEmbedding(self, c):
    raise NotImplementedError()

  def getBConceptHEmbedding(self, c):
    raise NotImplementedError()

  def getBConceptTEmbedding(self, c):
    raise NotImplementedError()
  

class TransE(EvalOnThisCodeModel):
  def __init__(self, model, triple_factory) -> None:
    super().__init__()
    self.model = model
    self.triple_factory = triple_factory
  
  def getEntityEmbedding(self, e):
    raise NotImplementedError()

  def getUConceptEmbedding(self, c):
    raise NotImplementedError()

  def getBConceptHEmbedding(self, c):
    raise NotImplementedError()

  def getBConceptTEmbedding(self, c):
    raise NotImplementedError()