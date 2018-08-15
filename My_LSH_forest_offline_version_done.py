#this is our implementation of LSH Forest
# We will be using synchronous way to find candidate points
import math
import random
import numpy as np
import gzip
import pickle
from sklearn.neighbors import NearestNeighbors
import operator
#with open('C:\Users\sony\Downloads\mnist.pkl','rb')as f:
with open('mnist.pkl','rb')as f:
    data = pickle.load(f)
data_size = input('Data_size = ')
L = input('L = ')
queries = range(2,2+input('# of Queries = '))
BF = input('Brute force = ')
leaf_size = input('max leaf_size = ')
NN = input('# of NN = ')
X = data[0][0]
X = X[range(0,data_size)]
test_data = data[2][0]
# normalization happens here
for i in range(data_size):
    mod =  np.linalg.norm(X[i])
    X[i] = X[i]/mod
for i in range(test_data.shape[0]):
    mod = np.linalg.norm(test_data[i])
    test_data[i] = test_data[i]/mod

print stop
class node:
    def __init__(self,points,level,left,right,parent,identity,hf):
        self.points = points
        self.level = level
        self.left = left
        self.right = right
        self.parent = parent
        self.identity = identity
        self.hf = np.transpose(hf)
        #self.hf = hf
def candi_to_final(candi_points,q,NN):
    global test_data
    global X
    if(len(candi_points)>=NN):
        array=[]
        for i in candi_points:
            array.append(np.linalg.norm(X[i]-test_data[q]))
        #print 'array = ',array
        final = []
        for i in range(len(array)):
            final.append([array[i],candi_points[i]])
        final = sorted(final)
        answer = []
        for i in range(NN):
            answer.append(int(final[i][1]))
        #print 'SVD ',answer
        return answer
    else:
        print error_aana_chahiye
    #    i = 1
     #   while(len(candi_points) < NN):
      #      candi_points.append(i)
        #    i = i + 1
       # return candi_points
frustrated = []
def maketree(counter,level,parent,arr):
    global X
    global frustrated
    global gcounter
    global linked_tree
    global unable_seperate
    # print np.mean(X[arr])
    if(len(linked_tree)-counter<10):
        linked_tree = linked_tree + [-1]*20
    if(len(arr)>leaf_size):
        hf = np.random.rand(784,1)-0.5
        arr1 = []
        arr2 = []
        for i in arr:
            if(np.dot(X[i],hf)>0):
                # goes to right
                arr1.append(i)
            else:
                # goes to left
                arr2.append(i)
        if(len(arr1)>0.25*len(arr) and len(arr2)>0.25*len(arr)):
            # print 'Was ablle to seperate',counter,unable_seperate
            unable_seperate = 0
            left = gcounter + 1
            right = gcounter +2
            gcounter = gcounter + 2
            linked_tree[counter] = node(arr,level,left,right,parent,counter,hf)
            maketree(left,level+1,counter,arr1)
            maketree(right,level+1,counter,arr2)
        else:
            # print 'It was not able to seperate',counter,unable_seperate
            if(unable_seperate>=10):
                linked_tree[counter] = node(arr,level,-1,-1,parent,counter,-1)
                unable_seperate = 0
#               print 'got frustrated of',counter
                frustrated.append(counter)
            else:
                unable_seperate = unable_seperate + 1
                maketree(counter,level,parent,arr)
    else:
        linked_tree[counter] = node(arr,level,-1,-1,parent,counter,-1)

linked_tree_array = [-1]*L
for i in range(L):
 #   print '####################'
  #  print '####################'
    gcounter = 0
    unable_seperate = 0
    linked_tree = [-1]*20
    maketree(0,0,-1,range(data_size))
    linked_tree_array[i] = linked_tree
'''    for j in range(len(linked_tree)):
        print '*****************'
        if(type(linked_tree[j])==type(1)):
            print linked_tree[j]
        else:
            print 'identity = ',linked_tree[j].identity
            print 'i points = ',linked_tree[j].points
            print 'left = ',linked_tree[j].left
            print 'right = ',linked_tree[j].right
            print 'level = ',linked_tree[j].level
            print 'parent = ',linked_tree[j].parent
            if (len(linked_tree[j].points)>leaf_size and linked_tree[j].left==-1 and linked_tree[j].right==-1 and linked_tree[j].identity not in frustrated):
                print stop'''
def jaccard(NN,approx,exact):
    t = len(list(set(exact.tolist()[0]).intersection(approx)))
    return float(t)/((2*NN)-t)
forest_dis_array = []
JS_forest = []
Exact_dis_array = []
BF_applied = []
for q in queries:
   # print '######################'
    probable_nodes = [-1]*L
    probable_levels = [-1]*L
    for i in range(L):
#       print 'Next Tree please'
        level = 0
        curr_node = linked_tree_array[i][0]
        while(True):
            if(level < 31):
                if(curr_node.right==-1 and curr_node.left == -1):
                    probable_nodes[i] = curr_node
                    probable_levels[i] = curr_node.level
                    break
                else:
                    if(np.dot(curr_node.hf,test_data[q])>0):
 #                       print 'Query wants to go right'
                        level = level + 1
                        curr_node = linked_tree_array[i][curr_node.right]
                    else:
  #                      print 'Query wants to go left'    
                        curr_node = linked_tree_array[i][curr_node.left]
                        level = level + 1
            else:
   #             print 'level>31'
                probable_nodes[i]=curr_node
                probable_levels[i] = curr_node.level
    #for w in probable_nodes:
#       print w.points,
 #   print 'probable levels = ',probable_levels
    candidates = set([])
    while(len(candidates)<BF):
        index, value = max(enumerate(probable_levels), key=operator.itemgetter(1))
        temp = candidates.union(probable_nodes[index].points)
        if(len(temp)>BF):
            candidates = temp
            break
        else:
            candidates = temp
            probable_nodes[index] = linked_tree_array[index][probable_nodes[index].parent]
            probable_levels[index] = probable_levels[index] - 1
    candidates = list(candidates)
#    print 'Candidates = ',candidates
    BF_applied.append(len(candidates))
    forest_answer = candi_to_final(candidates,q,NN)
  #  print 'answer = ',forest_answer

    nbrs = NearestNeighbors(algorithm='brute', metric='euclidean').fit(X)
    exact_neighbors = nbrs.kneighbors(test_data[q].reshape(1,784), n_neighbors=NN, return_distance=False)
   # print 'exact answer = ',exact_neighbors
    JS_forest.append(jaccard(NN,forest_answer,exact_neighbors))
    forest_dis = 0
    exact_dis = 0
    for i in forest_answer:
       forest_dis = forest_dis+np.linalg.norm(X[i]-test_data[q])
    for i in exact_neighbors[0]:
       exact_dis = exact_dis+np.linalg.norm(X[i]-test_data[q])
    forest_dis_array.append(forest_dis)
    Exact_dis_array.append(exact_dis)
print '/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*'
l = float(len(queries))
print 'AVG Brute force applied = ',sum(BF_applied)/l
print 'AVG forest Distance = ',sum(forest_dis_array)/l
print 'AVG Exact Dis = ',sum(Exact_dis_array)/l
print 'JS of forest = ',sum(JS_forest)/l
print 'Data points = ',data_size
print 'L = ',L
print 'queries = ',l
print 'BF = ',BF
print 'NN = ',NN



