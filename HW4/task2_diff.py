from collections import defaultdict
import sys
import time
import os
import itertools
from pyspark import SparkContext

#os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12")

def to_list(a):
    return [a]

def append(a, b):
    a.append(b)
    return a

def extend(a, b):
    a.extend(b)
    return a

def checkThreshold(user1, user2, threshold):
    if len(user1&user2) >= threshold:
        return True
    else:
        return False

def GirvanNewman(user_graph):

    def calculate_betweeness(root):
        levelOrder = {}
        parent_dict, child_dict = defaultdict(set), defaultdict(set)
        visited = set()
        visited.add(root)
        parentNo_dict = defaultdict(int)

        currentNodes = user_graph[root]
        level = 1

        levelOrder[0] = {root}
        child_dict[root]  = currentNodes

        while currentNodes:
            levelOrder[level] = currentNodes
            visited = visited.union(currentNodes)
            temp_nodes = set()

            for node in currentNodes:
                child_dict[node] = user_graph[node].difference(visited)
                temp_nodes = temp_nodes.union(child_dict[node])
            
            currentNodes = temp_nodes.difference(visited)
            level+=1

        node_credits, edge_credits = defaultdict(float), defaultdict(float)
        
        dag_edges = list()
        level-=1

        for i in range(level,0,-1):
            key =  i-1
            parent_nodes = levelOrder[key]
            
            for child_node in levelOrder[i]:
                actualParentNodes = user_graph[child_node].intersection(set(parent_nodes))  
                for parent in actualParentNodes:
                    dag_edges.append((child_node,parent))
        
        for child,parent in dag_edges:
            parent_dict[child].add(parent)
            child_dict[parent].add(child)

        shortest_paths = dict()
        shortest_paths[root] = 1
        
        for i in range(1,level+1):
            for child in levelOrder[i]:
                parents = parent_dict[child]
                no_of_paths = 0
                for parent in parents:
                    no_of_paths = no_of_paths + shortest_paths[parent]
                shortest_paths[child] = no_of_paths
        
        betweenness_dict = dict()

        for edge in dag_edges:
            parents =  parent_dict[edge[0]]
            if len(child_dict[edge[0]])==0:
                betweenness_dict[edge] = shortest_paths[edge[1]]/shortest_paths[edge[0]]
            else:
                credit = 1
                for child in child_dict[edge[0]]:
                    credit += betweenness_dict[(child,edge[0])]
                betweenness_dict[edge] = credit * shortest_paths[edge[1]]/shortest_paths[edge[0]]

        betweenness = [(sorted(k),v) for k,v in betweenness_dict.items()]
        return betweenness
    
    true_betweeness = defaultdict(list)
    for key in user_graph.keys():
        temp_list = calculate_betweeness(key)
        for k,v in temp_list:
            true_betweeness[tuple(k)].append(v)
    
    betweenness_list = list()
    for key,val in true_betweeness.items():
        true_value = sum(val) / 2
        betweenness_list.append((key,true_value))

    betweenness_list.sort(key = lambda x: (-x[1],x[0][0]))
    return betweenness_list

if __name__=='__main__':
    if len(sys.argv) !=5:
        print("All arguments haven't been specified")
        sys.exit(-1)

    threshold = int(sys.argv[1])
    input_file = sys.argv[2]
    betweenness_file = sys.argv[3]
    communities_file = sys.argv[4]

    sc= SparkContext('local[*]','Task1')
    sc.setLogLevel("WARN")

    start_time = time.time()

    train_file = sc.textFile(input_file)
    header = train_file.first()

    userID_businessID_dict = train_file.filter(lambda x: x!=header).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1])).combineByKey(to_list, append, extend).map(lambda x: (x[0], set(x[1]))).persist().collectAsMap()

    businessID_userID_dict = train_file.filter(lambda x: x!=header).map(lambda x: x.split(",")).map(lambda x: (x[1], x[0])).combineByKey(to_list, append, extend).map(lambda x: (x[0], set(x[1]))).persist().collectAsMap()

    userIDPairs = set()
    for users in businessID_userID_dict.values():
        for user1,user2 in list(itertools.combinations(users,2)):
            userIDPairs.add((user1,user2))
    
    vertices = set()
    edges = set()

    for userPair in userIDPairs:
        if(checkThreshold(set(userID_businessID_dict[userPair[0]]),set(userID_businessID_dict[userPair[1]]),threshold)):
            edges.add((userPair[0],userPair[1]))
            edges.add((userPair[1],userPair[0]))
            vertices.add(userPair[0])
            vertices.add(userPair[1])

    user_graph = defaultdict(set)

    for user1, user2 in edges:
        user_graph[user1].add(user2)

    betweeness_list = GirvanNewman(user_graph)
    with open(betweenness_file, 'w+') as fp:
        for i in betweeness_list:
            fp.writelines(str(i)[1:-1] + "\n")

    print(time.time()-start_time)