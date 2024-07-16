# purpose: model evaluation class - to load and evaluate an existing model on the test data and compute accuracy.

import os, sys
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# import torch_directml
import accuracy
from time import perf_counter


class ModelEval:
  def __init__(self, dataObj, entityCount, uConceptCount, bConceptCount, embedDim, batchSize, device, model):
    self.dataObj = dataObj
    self.entityCount = entityCount
    self.uConceptCount = uConceptCount
    self.bConceptCount = bConceptCount
    self.embedDim = embedDim
    self.batchSize = batchSize
    self.device = device
    self.model = model
    self.one = torch.FloatTensor([1.0])
    self.one = self.one.to(self.device)
    self.accObj = accuracy.Accuracy()

  def evalModel(self):
    self.computeEmbeddingQuality()

  def computeEmbeddingQuality(self):
    print('Embedding Quality:')
    # resU  = self.getUClassEmbeddingQuality(self.dataObj.aboxUCLst)
    # print(' > Train Unary Abox Classes')
    # print('   ', 'MR:', resU[0]/resU[4], 'MRR:', resU[1]/resU[4], 'Hit@1%:', resU[2]/resU[4], 'Hit@10%:', resU[3]/resU[4], 'Count:', resU[4])
    # resB = self.getBClassEmbeddingQuality(self.dataObj.aboxBCLst)
    # print(' > Train Binary Abox Classes')
    # print('   ', 'MR:', resB[0]/resB[4], 'MRR:', resB[1]/resB[4], 'Hit@1%:', resB[2]/resB[4], 'Hit@10%:', resB[3]/resB[4], 'Count:', resB[4])
    # print(' > Train All Abox Classes')
    # res = (resU[0]+resB[0], resU[1]+resB[1], resU[2]+resB[2], resU[3]+resB[3], resU[4]+resB[4])
    # print('   ', 'MR:', res[0]/res[4], 'MRR:', res[1]/res[4], 'Hit@1%:', res[2]/res[4], 'Hit@10%:', res[3]/res[4], 'Count:', res[4])

    resU = self.getTestUClassEmbeddingQuality(list(self.dataObj.testUCMemberMap.keys()))
    print(' > Test Unary Abox Classes')
    print('   ', 'MR:', resU[0]/resU[4], 'MRR:', resU[1]/resU[4], 'Hit@1%:', resU[2]/resU[4], 'Hit@10%:', resU[3]/resU[4], 'Count:', resU[4])
    resB = self.getTestBClassEmbeddingQuality(list(self.dataObj.testBCMemberMap.keys()))
    print(' > Test Binary Abox Classes')
    print('   ', 'MR:', resB[0]/resB[4], 'MRR:', resB[1]/resB[4], 'Hit@1%:', resB[2]/resB[4], 'Hit@10%:', resB[3]/resB[4], 'Count:', resB[4])
    print(' > Test All Abox Classes')
    res = (resU[0]+resB[0], resU[1]+resB[1], resU[2]+resB[2], resU[3]+resB[3], resU[4]+resB[4])
    print('   ', 'MR:', res[0]/res[4], 'MRR:', res[1]/res[4], 'Hit@1%:', res[2]/res[4], 'Hit@10%:', res[3]/res[4], 'Count:', res[4])


  def getUClassSpaceMembershipScore(self, uCE, eLst):
    raise Exception("deprecated use version2")
    uE = torch.LongTensor(eLst).to(self.device)
    uEE = self.model.getEntityEmbedding(uE)
    uCE = uCE.repeat(len(eLst), 1)
    tmp = (self.one-uCE)*uEE
    s = torch.sum(tmp*tmp, dim=1)
    sn = s.data.cpu().numpy()
    return sn

  def getBClassSpaceMembershipScore(self, bCE, eLst):
    raise Exception("deprecated use version2")
    bHE = []
    bTE = []
    for e in eLst:
      h, t = e
      bHE.append(h)
      bTE.append(t)
    bHE = Variable(torch.LongTensor(bHE))
    bTE = Variable(torch.LongTensor(bTE))
    bHE = bHE.to(self.device)
    bTE = bTE.to(self.device)
    bHEE = self.model.getEntityEmbedding(bHE)
    bHTE = self.model.getEntityEmbedding(bTE)
    bCHE, bCTE = bCE
    bCHE = bCHE.repeat(len(eLst), 1)
    bCTE = bCTE.repeat(len(eLst), 1)
    tmpH = (self.one-bCHE)*bHEE
    tmpT = (self.one-bCTE)*bHTE
    s = torch.sum(tmpH*tmpH, dim=1) + torch.sum(tmpT*tmpT, dim=1)
    sn = s.data.cpu().numpy()
    return sn

  def getUClassSpaceMembershipScore2(self, uCE, eLst):
    uE = torch.LongTensor(eLst).to(self.device)
    uEE = self.model.getEntityEmbedding(uE)
    uCE = (self.one - uCE).repeat(len(eLst), 1)
    tmp = uCE*uEE
    s = torch.sum(tmp*tmp, dim=1)
    return s
  
  def getBClassSpaceMembershipScore2(self, bCE, eLst):
    bHE, bTE = eLst
    len_elst = len(bHE)
    bHE = torch.LongTensor(bHE).to(self.device)
    bTE = torch.LongTensor(bTE).to(self.device)
    bHEE = self.model.getEntityEmbedding(bHE)
    bHTE = self.model.getEntityEmbedding(bTE)
    bCHE, bCTE = bCE
    bCHE = (self.one-bCHE).repeat(len_elst, 1)
    bCTE = (self.one-bCTE).repeat(len_elst, 1)
    tmpH = bCHE*bHEE
    tmpT = bCTE*bHTE
    s = torch.sum(tmpH*tmpH, dim=1) + torch.sum(tmpT*tmpT, dim=1)
    return s

  def getClassSpaceMembershipScore(self, cE, eLst):
    uHE = []
    bHE = []
    bTE = []
    uCount = 0
    bCount = 0
    eLstMap = []
    for e in eLst:
      h, t = e
      if t==None:
       uHE.append(h)
       eLstMap.append((0,uCount))
       uCount+=1
      else:
       bHE.append(h)
       bTE.append(t)
       eLstMap.append((1,bCount))
       bCount+=1
    uHE = Variable(torch.LongTensor(uHE))
    bHE = Variable(torch.LongTensor(bHE))
    bTE = Variable(torch.LongTensor(bTE))
    uHE = uHE.to(self.device)
    bHE = bHE.to(self.device)
    bTE = bTE.to(self.device)
    uHEE = self.model.getEntityEmbedding(uHE)
    uTEE = Variable(torch.FloatTensor(torch.zeros(len(uHE), self.embedDim)))
    uTEE = uTEE.to(self.device)
    uEE = torch.cat((uHEE, uTEE), 1)
    bHEE = self.model.getEntityEmbedding(bHE)
    bTEE = self.model.getEntityEmbedding(bTE)
    bEE = torch.cat((bHEE, bTEE), 1)
    eE = torch.cat((uEE, bEE), 0)
    cE = cE.repeat(len(eLst),1)
    tmp = (self.one-cE)*eE
    s = torch.sum(tmp*tmp, dim=1)
    sn = s.data.cpu().numpy()
    return sn

  def getTestUClassEmbeddingQuality(self, classLst):
    e2id = self.dataObj.getEntityMap()
    # uc2id = self.dataObj.getUConceptMap()
    entityLst = list(e2id.values())

    # allRanks = []
    allCandidateLstLen = 0
    allTrueMembersCount = 0
    mr = 0.0; mrr = 0.0; hit1 = 0.0; hit10 = 0.0; count = 0
    for c in classLst:
      testTrueMembers = set(self.dataObj.getTestUClassMembers(c))
      allTrueMembers = set(self.dataObj.getAllUClassMembers(c))
      # print(self.dataObj.id2uc[c], len(testTrueMembers))
      rank_device_list = []
      candidateLstLen = 0
      for testTrueMember in testTrueMembers:
        # Missing tail, unary predicate
        # candidate list must have true member as the first entry in the list, such that the position of argsorted score is the true member
        candidateLst = self.getUClassMembershipCandidateList(testTrueMember, allTrueMembers, entityLst)
        candidateLstLen += len(candidateLst)
        scoreLst = self.getUClassSpaceMembershipScore2(self.getUClassSpace(c), candidateLst)
        rankLst = self.accObj.getRankList2(scoreLst)
        rank_device_list.append(rankLst)

      ranks = [torch.nonzero(l==0, ).squeeze().item() + 1 for l in rank_device_list]
      allCandidateLstLen += candidateLstLen
      allTrueMembersCount += len(testTrueMembers)
      # print('   ', self.accObj.computeMetrics(ranks), candidateLstLen/len(testTrueMembers))
      res = self.accObj.computeMetrics(ranks)
      mr+=res['MR']; mrr+=res['MRR']; hit1+=res['R1%']; hit10+=res['R10%']; count+=1
    # print(allCandidateLstLen/allTrueMembersCount)
    return (mr, mrr, hit1, hit10, count)

  def getTestBClassEmbeddingQuality(self, classLst):
    e2id = self.dataObj.getEntityMap()
    # bc2id = self.dataObj.getBConceptMap()
    entityLst = list(e2id.values())

    allCandidateLstLen = 0
    allTrueMembersCount = 0
    mr = 0.0; mrr = 0.0; hit1 = 0.0; hit10 = 0.0; count = 0
    for c in classLst: 
      # start_time = perf_counter()
      testTrueMembers = set(self.dataObj.getTestBClassMembers(c))
      allTrueMembers = set(self.dataObj.getAllBClassMembers(c))
      
      rank_device_list = []
      candidateLstLen = 0
      for testTrueMember in testTrueMembers:
        # Missing head
        eLst, tLst = self.getBClassMembershipHCandidateList2(testTrueMember, allTrueMembers, entityLst)
        candidateLstLen += len(tLst)
        scoreLst = self.getBClassSpaceMembershipScore2(self.getBClassSpace(c), (eLst, tLst))
        hRankLst = self.accObj.getRankList2(scoreLst)
        rank_device_list.append(hRankLst)

        # Missing tail
        hLst, eLst = self.getBClassMembershipTCandidateList2(testTrueMember, allTrueMembers, entityLst)
        candidateLstLen += len(hLst)
        scoreLst = self.getBClassSpaceMembershipScore2(self.getBClassSpace(c), (hLst, eLst))
        tRankLst = self.accObj.getRankList2(scoreLst)
        rank_device_list.append(tRankLst)
        
      ranks = [torch.nonzero(l==0, ).squeeze().item() + 1 for l in rank_device_list]
      allCandidateLstLen += candidateLstLen
      allTrueMembersCount += 2*len(testTrueMembers)
      res = self.accObj.computeMetrics(ranks)
      mr+=res['MR']; mrr+=res['MRR']; hit1+=res['R1%']; hit10+=res['R10%']; count+=1

      # end_time = perf_counter()
      # print(f'\telapsed {end_time-start_time:.3f}s',self.accObj.computeMetrics(ranks), candidateLstLen/(2*len(testTrueMembers)))
    # print(allCandidateLstLen/allTrueMembersCount)
      
    return (mr, mrr, hit1, hit10, count)

  def getUClassEmbeddingQuality(self, classLst):
    e2id = self.dataObj.getEntityMap()
    # uc2id = self.dataObj.getUConceptMap()
    entityLst = list(e2id.values())

    # allRanks = []
    allCandidateLstLen = 0
    allTrueMembersCount = 0
    mr = 0.0; mrr = 0.0; hit1 = 0.0; hit10 = 0.0; count = 0
    for c in classLst:
      trueMembers = set(self.dataObj.getUClassMembers(c))
      print(self.dataObj.id2uc[c], len(trueMembers))
      rank_device_list = []
      candidateLstLen = 0
      for trueMember in trueMembers:
        candidateLst = self.getUClassMembershipCandidateList(trueMember, trueMembers, entityLst)
        candidateLstLen += len(candidateLst)
        scoreLst = self.getUClassSpaceMembershipScore2(self.getUClassSpace(c), candidateLst)
        rankLst = self.accObj.getRankList2(scoreLst)
        rank_device_list.append(rankLst)
        # rank = numpy.where(rankLst==0)[0][0] + 1
        # ranks.append(rank)
      allCandidateLstLen += candidateLstLen
      allTrueMembersCount += len(trueMembers)
      ranks = [torch.nonzero(l==0, ).squeeze().item() + 1 for l in rank_device_list]
      # allRanks.append(ranks)
      print('   ', res:=self.accObj.computeMetrics(ranks), candidateLstLen/len(trueMembers))
      mr+=res['MR']; mrr+=res['MRR']; hit1+=res['R1%']; hit10+=res['R10%']; count+=1
    print(allCandidateLstLen/allTrueMembersCount)
    return (mr, mrr, hit1, hit10, count)

  def getBClassEmbeddingQuality(self, classLst):
    e2id = self.dataObj.getEntityMap()
    bc2id = self.dataObj.getBConceptMap()
    entityLst = list(e2id.values())

    allRanks = []
    allCandidateLstLen = 0 
    allTrueMembersCount = 0
    mr = 0.0; mrr = 0.0; hit1 = 0.0; hit10 = 0.0; count = 0
    for c in classLst:
      trueMembers = set(self.dataObj.getBClassMembers(c))
      print(self.dataObj.id2bc[c], len(trueMembers))
      ranks = []
      candidateLstLen = 0
      for trueMember in trueMembers:
        candidateLst = self.getBClassMembershipCandidateList(trueMember, trueMembers, entityLst)
        candidateLstLen += len(candidateLst)
        scoreLst = self.getBClassSpaceMembershipScore2(self.getBClassSpace(c), candidateLst)
        rankLst = self.accObj.getRankList(scoreLst)
        rank = numpy.where(rankLst==0)[0][0] + 1
        ranks.append(rank)
        allRanks.append(rank)
      allCandidateLstLen += candidateLstLen
      allTrueMembersCount += len(trueMembers)
      print('   ',self.accObj.computeMetrics(ranks), candidateLstLen/len(trueMembers))
      res = self.accObj.computeMetrics(ranks)
      mr+=res['MR']; mrr+=res['MRR']; hit1+=res['R1%']; hit10+=res['R10%']; count+=1
    print(allCandidateLstLen/allTrueMembersCount)
    return (mr, mrr, hit1, hit10, count)




  def getUClassMembershipCandidateList(self, trueMember, trueMembers, entityLst):
    # candidate list must have true member as the first entry in the list
    candidateLst = [trueMember]
    for e in entityLst:
      if e not in trueMembers:
        candidateLst.append(e)
    return candidateLst

  def getBClassMembershipCandidateList(self, trueMember, trueMembers, entityLst):
    candidateLst = [trueMember]
    h, t = trueMember
    for e in entityLst:
      if not (h, e) in trueMembers:
        candidateLst.append((h, e))
      if not (e, t) in trueMembers:
        candidateLst.append((e, t))
    return candidateLst

  def getBClassMembershipHCandidateList(self, trueMember, trueMembers, entityLst):
    candidateLst = []
    candidateLst.append(trueMember)
    h, t = trueMember
    for e in entityLst:
      if not (e, t) in trueMembers:
        candidateLst.append((e, t))
    return candidateLst

  def getBClassMembershipHCandidateList2(self, trueMember, trueMembers, entityLst):
    h, t = trueMember
    eLst = [h]
    tLst = [t]
    for e in entityLst:
      if not (e, t) in trueMembers:
        eLst.append(e)
        tLst.append(t)
    return eLst, tLst

  def getBClassMembershipTCandidateList(self, trueMember, trueMembers, entityLst):
    candidateLst = []
    candidateLst.append(trueMember)
    h, t = trueMember
    for e in entityLst:
      if not (h, e) in trueMembers:
        candidateLst.append((h, e))
    return candidateLst
  
  def getBClassMembershipTCandidateList2(self, trueMember, trueMembers, entityLst):
    h, t = trueMember
    hLst = [h]
    eLst = [t]
    for e in entityLst:
      if not (h, e) in trueMembers:
        hLst.append(h)
        eLst.append(e)
    return hLst, eLst

  def getUClassSpace(self, c):
    cT = torch.LongTensor([c])
    cT = cT.to(self.device)
    uCE = self.model.getUConceptEmbedding(cT)
    return uCE

  def getBClassSpace(self, c):
    cT = torch.LongTensor([c])
    cT = cT.to(self.device)
    bCHE = self.model.getBConceptHEmbedding(cT)
    bCTE = self.model.getBConceptTEmbedding(cT)
    return bCHE, bCTE

  def getEntityEmbedding(self, eName):
    eT = Variable(torch.LongTensor([self.dataObj.getEntityId(eName)]))
    eT = eT.to(self.device)
    eE = self.model.getEntityEmbedding(eT)
    return eE

  def getClassEmbedding(self, cName):
    return self.getClassSpace(self.dataObj.getClassId(cName))

  def getAccuracyPrintText(self, resObj):
    retVal = 'MR='+'{:.1f}'.format(resObj['MR'])
    retVal += ', MRR='+'{:.2f}'.format(resObj['MRR'])
    retVal += ', R1%='+'{:.1f}'.format(resObj['R1%'])
    retVal += ', R2%='+'{:.1f}'.format(resObj['R2%'])
    retVal += ', R3%='+'{:.1f}'.format(resObj['R3%'])
    retVal += ', R5%='+'{:.1f}'.format(resObj['R5%'])
    retVal += ', R10%='+'{:.1f}'.format(resObj['R10%'])
    return retVal





  def getSortedKeyList(self, keyValueMap):
    keyLst = list(keyValueMap.keys())
    i=0
    while i<len(keyLst):
      j=i+1
      while j<len(keyLst):
        if keyValueMap[keyLst[i]]>keyValueMap[keyLst[j]]:
          tmp = keyLst[i]
          keyLst[i] = keyLst[j]
          keyLst[j] = tmp
        j+=1
      i+=1
    return keyLst
  