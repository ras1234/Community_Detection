#!/usr/bin/env python
import networkx as nx
import math
import csv
import random as rand
import sys

with open("auth_id.txt") as f:
    content = f.readlines()

author = {}
for row in content:
    tem = row.split("#")
    dum = tem[1].replace("\n", "")
    author[dum] = tem[0]

def get_author(author_id):  
    return author[str(author_id)]
target = open("community.txt",'w')
target.truncate()
test=nx.Graph()
#reading graph structure from the file
def buildG(G, file_, delimiter_):
    global test
    reader = csv.reader(open(file_), delimiter=delimiter_)
    for line in reader:
        if len(line) > 2:
            if float(line[2]) != 0.0:
                try:
                    a = int(line[0])
                    b = int(line[1])
                    c = float(line[2])
                except ValueError:
                    continue
                else:
                    G.add_edge(int(line[0]),int(line[1]),weight=float(line[2]))
        else:
            G.add_edge(int(line[0]),int(line[1]),weight=1.0)
    return G
#keep removing edges from Graph until one of the connected components of Graph splits into two
#compute the edge betweenness
def CmtyGirvanNewmanStep(G):
    init_ncomp = nx.number_connected_components(G)    #no of components
    ncomp = init_ncomp
    while ncomp <= init_ncomp:
        bw = nx.edge_betweenness_centrality(G, weight='weight')    #edge betweenness for G
        #find the edge with max centrality
        max_ = max(bw.values())
        #find the edge with the highest centrality and remove all of them if there is more than one!
        for k, v in bw.iteritems():
            if float(v) == max_:
                G.remove_edge(k[0],k[1])    #remove the central edge
        ncomp = nx.number_connected_components(G)    #recalculate the no of components

#compute the modularity of current split
def _GirvanNewmanGetModularity(G, deg_, m_):
    New_A = nx.adj_matrix(G)
    New_deg = {}
    New_deg = UpdateDeg(New_A, G.nodes())
    #Let's compute the Q
    comps = nx.connected_components(G)    #list of components    
    print 'no of comp: %d' % nx.number_connected_components(G)
    Mod = 0    #Modularity of a given partitionning
    for c in comps:
        EWC = 0    #no of edges within a community
        RE = 0    #no of random edges
        for u in c:
            EWC += New_deg[u]
            RE += deg_[u]        #count the probability of a random edge
        Mod += ( float(EWC) - float(RE*RE)/float(2*m_) )
    Mod = Mod/float(2*m_)
    return Mod

def UpdateDeg(A, nodes):
    deg_dict = {}
    n = len(nodes)
    B = A.sum(axis = 1)
    for i in range(n):
        deg_dict[nodes[i]] = B[i, 0]
    return deg_dict

#run GirvanNewman algorithm and find the best community split by maximizing modularity measure
def runGirvanNewman(G, Orig_deg, m_):
    #find the best split of the graph
    BestQ = 0.0
    Q = 0.0
    test = nx.Graph()
    while True:   
        CmtyGirvanNewmanStep(G)
        Q = _GirvanNewmanGetModularity(G, Orig_deg, m_);
        print "current modularity: %f" % Q
        if Q > BestQ:
            BestQ = Q
            test = G.copy()
            Bestcomps = nx.connected_components(G)    #Best Split
        if G.number_of_edges() == 0:
            return test
    
    

def main():
    global otp
    graph_fn = "graphSmallest.txt"
    G = nx.Graph() 
    G = buildG(G, graph_fn, ',')
    n = G.number_of_nodes() 
    A = nx.adj_matrix(G)   
    m_ = 0.0    #the weighted version for number of edges
    for i in range(0,n):
        for j in range(0,n):
            m_ += A[i,j]
    m_ = m_/2.0
    print "m: %f" % m_

    #calculate the weighted degree for each node
    Orig_deg = {}
    Orig_deg = UpdateDeg(A, G.nodes())
    G = runGirvanNewman(G, Orig_deg, m_)
    otp = G
    nodes = otp.nodes()
    while len(nodes) > 0:
        community = []
        neighbours = otp.neighbors(nodes[0])
        community.append(nodes[0])
        nodes.remove(nodes[0])
        while len(neighbours) > 0:
            if neighbours[0]  in nodes:
                community.append(neighbours[0])
                nodes.remove(neighbours[0])
            tem = otp.neighbors(neighbours[0])
            for item in tem:
                if item in nodes:
                    neighbours.append(item)
            neighbours.remove(neighbours[0])
        print "writting community:"
        for items in community:
            target.write(str(get_author(items))+",")
        target.write("\n")
main()
