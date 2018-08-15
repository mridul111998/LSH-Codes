import math
import random
import numpy as np
import gzip
import pickle
import operator
from sklearn.neighbors import NearestNeighbors
import operator
with open('mnist.pkl','rb')as f:
    data = pickle.load(f)

data_size = input('Data_size = ')
leaf_size = input('Leaf_size = ')
queries = range(2,2+input('# of Queries = '))
#BF = input('Brute force = ')
NN = input('# of NN = ')
threshold = input('Threshold = ')

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

class node:
    def __init__(self,points,level,left,right,parent,identity,hf):
        self.points = points
        self.level = level
        self.left = left
        self.right = right
        self.parent = parent
        self.identity = identity
        self.hf = np.transpose(hf)

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
        print error_dena_chahiye


not_seperated = []
# not_seprated are those leaves which were not seperated even after certain number of random hyperplanes
def maketree(counter,level,parent,arr):
    global X
    global gcounter
    global linked_tree
    global unable_seperate
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
       #     print 'Was able to seperate ',counter,unable_seperate
            unable_seperate = 0
            left = gcounter + 1
            right = gcounter +2
            gcounter = gcounter + 2
            linked_tree[counter] = node(arr,level,left,right,parent,counter,hf)
            maketree(left,level+1,counter,arr1)
            maketree(right,level+1,counter,arr2)
        else:
            # It was not able to seperate
            if(unable_seperate>=10):
       #         print 'given up on ',counter
                not_seperated.append(counter)
                linked_tree[counter] = node(arr,level,-1,-1,parent,counter,-1)
                unable_seperate = 0
            else:
        #        print 'unable to seperate',counter
                unable_seperate = unable_seperate + 1
                maketree(counter,level,parent,arr)
    else:
        # size of leaf is less than leaf_size
        linked_tree[counter] = node(arr,level,-1,-1,parent,counter,-1)

#print '####################'
gcounter = 0
unable_seperate = 0
linked_tree = [-1]*20
maketree(0,0,-1,range(data_size))
'''for j in range(len(linked_tree)):
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

for i in not_seperated:
    print 'identity = ',linked_tree[i].identity
    print 'i points = ',linked_tree[i].points
'''
def jaccard(NN,approx,exact):
    t = len(list(set(exact.tolist()[0]).intersection(approx)))
    return float(t)/((2*NN)-t)

def query(q,counter,th):
    # q is the query point
    # counter is the nodes under consideration
    # th is the threshold used uptil now
    global threshold
    global linked_tree
    global test_data
    curr_node = linked_tree[counter]
    if(th<=threshold):
        if(curr_node.right==-1 and curr_node.left==-1):
   #         print counter,' is a leaf'
            probable_nodes.append(curr_node)
        else:
            if(np.dot(curr_node.hf,test_data[q])>0):
    #            print 'preferable direction from',counter,'is right'
                query(q,curr_node.right,th)
                query(q,curr_node.left,th+1)
            else:
     #           print 'preferable direction from',counter,'is left'
                query(q,curr_node.left,th)
                query(q,curr_node.right,th+1)
   # else:
    #    print counter,'ki kahani khatam'
candi_sizes = []
dis_array = []
JS_array = []
Exact_dis_array = []
for q in queries:
    probable_nodes = []
    query(q,0,0)
  #  for i in probable_nodes:
   #     print '###########'
    #    print i.points
    l = []
    for i in probable_nodes:
        l = l+i.points
    probable_nodes = l
  #  print '***********'
   # print 'candidates = ',probable_nodes
   # print 'candidate_size  = ',len(probable_nodes)
    candi_sizes.append(len(probable_nodes))
    candidates = probable_nodes
    answer = candi_to_final(candidates,q,NN)
   # print 'answer = ',answer
    nbrs = NearestNeighbors(algorithm='brute', metric='euclidean').fit(X)
    exact_neighbors = nbrs.kneighbors(test_data[q].reshape(1,784), n_neighbors=NN, return_distance=False)
   # print 'exact answer = ',exact_neighbors
    JS_array.append(jaccard(NN,answer,exact_neighbors))
    dis = 0
    exact_dis = 0
    for i in answer:
       dis = dis+np.linalg.norm(X[i]-test_data[q])
    for i in exact_neighbors[0]:
       exact_dis = exact_dis+np.linalg.norm(X[i]-test_data[q])
    dis_array.append(dis)
    Exact_dis_array.append(exact_dis)

print '/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*'
l = float(len(queries))
print 'AVG Distance = ',sum(dis_array)/l
print 'AVG Exact Dis = ',sum(Exact_dis_array)/l
print 'JS = ',sum(JS_array)/l
print 'AVG candidates per query = ',sum(candi_sizes)/l
print 'data_size = ',data_size
print '# of Queries = ',len(queries)
print 'leaf_size = ',leaf_size
print '# of NN = ',NN
print 'Threshold = ',threshold


