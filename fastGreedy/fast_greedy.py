#!/usr/bin/python
#coding:utf-8

import sys
import copy

def fast_greedy(data):
  
    #community attribute
    com_v = []
    com_Q = []

    #vertice attribute
    ver_degree = {}
    ver_com = {}

    #edge attribute
    edge_com = {}

    m = len(data)
    m2 = m*2
    Q = 0
    QQ_max = 0
    COM_v_max = 0

    #initialize
    for v1,v2 in data:
        #store degree of vertices
        ver_degree[v1] = ver_degree.get(v1,0)+1
        ver_degree[v2] = ver_degree.get(v2,0)+1
	#store astription of edges
        edge_com[(v1,v2)] = -1

    for (index,v) in enumerate(ver_degree.keys()):
        #initialize astription of vertices
        ver_com[v] = index
	#initial vertices of community
        com_v.append(set([v]))
	#initial Q value of community
        com_Q.append(0)

    #main loop
    while True:
        #no extra edges any more
        if len(data) == 0:
            break
        maxQ = -999999
        maxIndex = -1
	#iterate over remaining edges
	#find the one with highest delta Q
        for index,(v1,v2) in enumerate(data):
            com1 = ver_com[v1]
            com2 = ver_com[v2]
            new_e = set()
            tmpQ = 0
            #edges between com1 and com2 are all fresh
            #in the combined community
            for vlast1 in com_v[com1]:
                for vlast2 in com_v[com2]:
		    #edge exists
                    if (vlast1,vlast2) in edge_com or (vlast2,vlast1) in edge_com:
                        tmpQ += m2-ver_degree[vlast1]*ver_degree[vlast2]
		    #edge not exists
                    else:
                        tmpQ += -ver_degree[vlast1]*ver_degree[vlast2]
            print tmpQ,maxQ
            if tmpQ > maxQ:
                maxQ = tmpQ
                maxIndex = index
	#for too big network, where degree[v1]*degree[v2]>999999
        if maxQ == -999999:
            break

        #combine com1 and com2
        v1,v2 = data[maxIndex]
        com1 = ver_com[v1]
        com2 = ver_com[v2]
        for vlast1 in com_v[com1]:
            for vlast2 in com_v[com2]:
                if (vlast1,vlast2) in edge_com:
                    edge_com[(vlast1,vlast2)] = com1
                if (vlast2,vlast1) in edge_com:
                    edge_com[(vlast2,vlast1)] = com1
        com_v[com1] = com_v[com1].union(com_v[com2])
        com_Q[com1] = com_Q[com1]+com_Q[com2]+maxQ
        com_Q[com2] = 0
        for tmpv in com_v[com2]:
            ver_com[tmpv] = com1
        com_v[com2].clear()

	#record info of community with biggest Q
        Q+=maxQ
        if Q > QQ_max:
            QQ_max = Q
            COM_v_max = copy.deepcopy(com_v)

	#delete edges between com1 and com2 from data
        datacopy = []
        for v1,v2 in data:
            if edge_com[(v1,v2)] == -1:
                datacopy.append((v1,v2))
        data = datacopy

    #result
    index = 0
    ret = []
    #print COM_v_max
    for community in COM_v_max:
        if len(community) > 0:/
            index += 1
            for vertice in community:
                ret.append((str(index),vertice))
        else:
            continue
    return ret
