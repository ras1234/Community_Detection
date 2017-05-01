#!/usr/bin/python
#coding:utf-8

import sys
from fast_greedy import fast_greedy

with open('data.txt','r') as f:
    data = []
    for line in f.readlines():
    	print line
        v1,v2 = line.strip().split(" ")
        data.append((v1,v2))
    print "out"
    #print data
    ret = fast_greedy(data)
    print "back"
    for x in ret:
        print "\t".join(x)
