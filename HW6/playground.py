import math
from sklearn.cluster import KMeans
import numpy as np
import sys
import time

def updatedClusters(kMeansClusters):
    temp = dict()
    for i in range(len(kMeansClusters)):
        clusterid = kMeansClusters[i]
        if clusterid in temp:
            temp[clusterid].append(i)
        else:
            temp[clusterid] = [i]
    return temp

def intermediateOuput(round_no):
    global ds_set
    global cs_dict
    global intermediate_output
    global rs_set

    ds_points, cs_clusters, cs_points = 0,0,0

    for _,val in ds_set.items():
        ds_points += val[1]
    
    for _, val in cs_dict.items():
        cs_clusters += 1
        cs_points += val[1]

    tempString = "Round " + str(round_no) + ": " + str(ds_points) + "," + str(cs_clusters) + "," + str(cs_points) + "," + str(len(rs_set)) + "\n"
    intermediate_output += tempString

def createDS(key, indices, points):
    global ds_set
    global pctrIdx_dict

    ds_set[key] = {}
    ds_set[key][0] = []

    for idx in indices:
        ds_set[key][0].append(pctrIdx_dict[idx])
    
    ds_set[key][1] = len(ds_set[key][0])
    ds_set[key][2] = np.sum(points[indices, :].astype(float), axis = 0)
    ds_set[key][3] = np.sum((points[indices, :].astype(float)) ** 2, axis = 0)
    ds_set[key][4] = np.sqrt((ds_set[key][3][:] / ds_set[key][1]) - (np.square(ds_set[key][2][:]) / (pow(ds_set[key][1],2))))
    ds_set[key][5] = ds_set[key][2] / ds_set[key][1]


def createCS(key, indices, points):
    global cs_dict
    global rs_dict
    global rs_set

    cs_dict[key] = {}
    cs_dict[key][0] = []

    for idx in indices:
        pointID = list(rs_dict.keys())[list(rs_dict.values()).index(rs_set[idx])]
        cs_dict[key][0].append(pointID)
    
    cs_dict[key][1] = len(cs_dict[key][0])
    cs_dict[key][2] = np.sum(points[indices, :].astype(float), axis = 0)
    cs_dict[key][3] = np.sum(pow(points[indices, :].astype(float),2), axis = 0)
    cs_dict[key][4] = np.sqrt((cs_dict[key][3][:] / cs_dict[key][1]) - (np.square(cs_dict[key][2][:]) / (pow(cs_dict[key][1],2))))
    cs_dict[key][5] = cs_dict[key][2] / cs_dict[key][1]

def getNearestClusterID(point, dictionary):
    global threshold
    global dimensions

    min_dist = threshold
    min_id = -1

    for key, val in dictionary.items():
        stdDev = val[4].astype(float)
        centroid = val[5].astype(float)
        md_dist = 0
        for d in range(dimensions):
            md_dist += pow(((point[d] - centroid[d]) / stdDev[d]),2)
        md_dist = np.sqrt(md_dist)

        if md_dist<min_dist:
            min_dist = md_dist
            min_id = key 
    
    return min_id

def updateDictionaries(summary, index, point, clusterID):
    global dimensions

    summary[clusterID][0].append(index)
    summary[clusterID][1] +=1
    for d in range(dimensions):
        summary[clusterID][2][d] = summary[clusterID][2][d] + point[d]
        summary[clusterID][3][d] = summary[clusterID][3][d] + pow(point[d],2)
    summary[clusterID][4] = np.sqrt((summary[clusterID][3][:] / summary[clusterID][1]) - (np.square(summary[clusterID][2][:]) / (pow(summary[clusterID][1],2))))
    summary[clusterID][5] = summary[clusterID][2] / summary[clusterID][1]

def getNearestClusterDict(firstSummary,secondSummary):
    global threshold
    nearestClusterDict = dict()

    for firstKey in firstSummary.keys():
        minDist = threshold
        minID = firstKey
        for secondKey in secondSummary.keys():
            if firstKey!=secondKey:
                firstStdDev = firstSummary[firstKey][4]
                firstCentroid = firstSummary[firstKey][5]
                secondStdDev = secondSummary[secondKey][4]
                secondCentroid = secondSummary[secondKey][5]
                firstMD = 0
                secondMD = 0

                for d in range(dimensions):
                    if firstStdDev[d]!=0 and secondStdDev[d]!=0:
                        firstMD =  firstMD + pow(((firstCentroid[d] -  secondCentroid[d]) / secondStdDev[d]),2)
                        secondMD =  secondMD + pow(((secondCentroid[d] -  firstCentroid[d]) / firstStdDev[d]),2)
                MD_Dist = min(np.sqrt(firstMD), np.sqrt(secondMD))
                if MD_Dist < minDist:
                    minDist = MD_Dist
                    minID = secondKey
        nearestClusterDict[firstKey] = minID
    
    return nearestClusterDict

if __name__=='__main__':
    if len(sys.argv) !=4:
        print("All arguments haven't been specified")
        sys.exit(-1)

    start_time = time.perf_counter()
    
    input_file = sys.argv[1]
    no_of_clusters = int(sys.argv[2])
    output_file = sys.argv[3]

    intermediate_output = "The intermediate results:\n"

    input_data = np.loadtxt(input_file, delimiter=",").tolist()

    input_size = len(input_data)
    chunk_size = int(input_size * 0.2)

    init_data = input_data[0:chunk_size]

    init_points = list()
    pointIdx_dict = {}
    pctrIdx_dict = {}
    
    temp = 0
    for data in init_data:
        idx,point = data[0], data[2:]
        init_points.append(point)
        pctrIdx_dict[temp] = idx
        pointIdx_dict[str(point)] = idx
        temp+=1

    dimensions = len(init_points[0])    
    threshold = 2 * math.sqrt(dimensions)
    npPoints = np.array(init_points)

    kLargeMeans = KMeans(n_clusters=5*no_of_clusters, random_state=553).fit_predict(npPoints)
    clusters = dict()

    for i in range(len(kLargeMeans)):
        if kLargeMeans[i] in clusters:
            clusters[kLargeMeans[i]].append(init_points[i])
        else:
            clusters[kLargeMeans[i]] = [init_points[i]]
    
    rs_dict = {}

    for key,val in clusters.items():
        if len(val)==1:
            point = val[0]
            idx = init_points.index(point)
            rs_dict[pctrIdx_dict[idx]] = point
            init_points.remove(point)
            for i in range(idx, len(pctrIdx_dict) - 1):
                pctrIdx_dict[i] = pctrIdx_dict[i+1]

    npActualPoints = np.array(init_points)
    kOrgMeans = KMeans(n_clusters=no_of_clusters, random_state=553).fit_predict(npActualPoints)

    clusters = updatedClusters(kOrgMeans)

    ds_set = dict()

    for k,v in clusters.items():
        createDS(k, clusters[k], npActualPoints)

    rs_set = list()

    for key,val in rs_dict.items():
        rs_set.append(val)

    npRSPoints = np.array(rs_set)
    kMeansRS = KMeans(n_clusters=int(len(rs_set)*0.6), random_state=553).fit_predict(npRSPoints)

    csClusters = updatedClusters(kMeansRS)

    cs_dict = {}

    for key,val in csClusters.items():
        if(len(val)>1):
            createCS(key, csClusters[key], npRSPoints)

    for key,val in csClusters.items():
        if(len(val)>1):
            for i in val:
                removePoint = list(rs_dict.keys())[list(rs_dict.values()).index(rs_set[i])]
                del rs_dict[removePoint]
        
    rs_set = list()
    for key,val in rs_dict.items():
        rs_set.append(val)

    intermediateOuput(1)
    end_idx = chunk_size
    for no_round in range(4):
        start_idx = end_idx
        if no_round==3:
            end_idx = len(input_data)
        else:
            end_idx = start_idx + chunk_size
        
        new_data = input_data[start_idx:end_idx]

        point_data = list()
        last_temp = temp

        for data in new_data:
            idx = data[0]
            point = data[2:]
            point_data.append(point)
            pctrIdx_dict[temp] = idx
            pointIdx_dict[str(point)] = idx
            temp+=1

        npPointsArr = np.array(point_data)

        for i in range(len(npPointsArr)):
            point = npPointsArr[i].astype(float)
            idx = pctrIdx_dict[last_temp+i]
            nearestClusterID = getNearestClusterID(point, ds_set)

            if nearestClusterID>-1:
                updateDictionaries(ds_set, idx, point, nearestClusterID)
            else:
                nearestClusterID = getNearestClusterID(point, cs_dict)
                if nearestClusterID > -1:
                    updateDictionaries(cs_dict, idx, point, nearestClusterID)
            
                else:
                    rs_dict[idx] = npPointsArr[i].tolist()
                    rs_set.append(npPointsArr[i].tolist())
                

        npPointsArr = np.array(rs_set)
        kMeans = KMeans(n_clusters=int(len(rs_set) * 0.6), random_state=553).fit_predict(npPointsArr)

        csClusters =  updatedClusters(kMeans)

        for key,value in csClusters.items():
            if len(value) > 1:
                temp_key = 0
                if key in cs_dict.keys():
                    while temp_key in cs_dict:
                        temp_key += 1
                else:
                    temp_key = key
                createCS(temp_key, csClusters[key],npPointsArr)
        
        for key,value in csClusters.items():
            if len(value)>1:
                for k in csClusters[key]:
                    removePoint = pointIdx_dict[str(rs_set[k])]
                    if removePoint in rs_dict.keys():
                        del rs_dict[removePoint]
        
        rs_set = list()
        for key, val in rs_dict.items():
            rs_set.append(val)

        closestClusterDict = getNearestClusterDict(cs_dict, cs_dict)

        for key,val in closestClusterDict.items():
            if key!= val and val in cs_dict.keys() and key in cs_dict.keys():
                firstKey, secondKey = key, val
                cs_dict[firstKey][0].extend(cs_dict[secondKey][0])
                cs_dict[firstKey][1] += cs_dict[secondKey][1]

                for d in range(dimensions):
                    cs_dict[firstKey][2][d] = cs_dict[firstKey][2][d] + cs_dict[secondKey][2][d]
                    cs_dict[firstKey][3][d] = cs_dict[firstKey][3][d] + cs_dict[secondKey][3][d]

                cs_dict[firstKey][4] = np.sqrt((cs_dict[firstKey][3][:] / cs_dict[firstKey][1]) - (np.square(cs_dict[firstKey][2][:]) / (pow(cs_dict[firstKey][1], 2))))
                cs_dict[firstKey][5] = cs_dict[firstKey][2] / cs_dict[firstKey][1]
                del cs_dict[closestClusterDict[key]]

        if no_round==3:
            closestClusterDict = getNearestClusterDict(cs_dict, ds_set)
            for key, value in closestClusterDict.items():
                if value in ds_set.keys() and key in cs_dict.keys():
                    csKey, dsKey = key, value
                    ds_set[dsKey][0].extend(cs_dict[csKey][0])
                    ds_set[dsKey][1] = ds_set[dsKey][1] + cs_dict[csKey][1]
                    
                    for d in range(dimensions):
                        ds_set[dsKey][2][d] = ds_set[dsKey][2][d] + cs_dict[csKey][2][d]
                        ds_set[dsKey][3][d] = ds_set[dsKey][3][d] + cs_dict[csKey][3][d]
                    
                    ds_set[dsKey][4] = np.sqrt((ds_set[dsKey][3][:] / ds_set[dsKey][1]) - (np.square(ds_set[dsKey][2][:]) / (pow(ds_set[dsKey][1],2))))
                    ds_set[dsKey][5] = ds_set[dsKey][2] / ds_set[dsKey][1]
                    del cs_dict[key]

        intermediateOuput(no_round+2)

    pointCluster_dict = dict()

    for key, val in ds_set.items():
        for point in val[0]:
            pointCluster_dict[point] = key
    for key, val in cs_dict.items():
        for point in val[0]:
            pointCluster_dict[point] = -1
    for point in rs_dict:
        pointCluster_dict[point] = -1
    
    sortedPointCluster = sorted(pointCluster_dict.keys(), key=int)

    with open(output_file,"w") as f:
        f.write(intermediate_output)
        f.write("\n")
        f.write("The clustering results:\n")
        for point in sortedPointCluster:
            f.write(str(int(point))+","+str(pointCluster_dict[point])+"\n")

    print(time.perf_counter()-start_time)



