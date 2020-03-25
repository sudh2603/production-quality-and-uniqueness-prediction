# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:12:34 2018

@author: sudhanshu kumar sinh
"""

import numpy as np
import random 

def majorvote(votes):
    vote_count={}
    for vote in votes:
        if vote in vote_count:
            vote_count[vote]+=1
        else:
            vote_count[vote]=1
    
    max_vote=max(vote_count.values())
    winner=[]
    for vote,count in vote_count.items():
        if count==max_vote:
            winner.append(vote)
    
    return random.choice(winner)

def distance(p1,p2):
    return np.sqrt(np.sum(np.power(p1-p2,2)))

def nearest_neighbour(p,points,k):

    distances=np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i]=distance(p,points[i])
    ind=np.argsort(distances)
    return ind[:k]

def knn_predict(p,points,outcomes,k):
    ind=nearest_neighbour(p,points,k)
    return majorvote(outcomes[ind])