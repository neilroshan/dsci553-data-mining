import sys
import time
import numpy as np
import math
from sklearn.cluster import KMeans


SAMPLE_PERCENTAGE = 0.2

N_INDEX = 0
SUM_INDEX = 1
SUMSQ_INDEX = 2
CENTROID_INDEX = 3
VARIANCE_INDEX = 4
STD_DEV_INDEX = 5
ROW = 1
COLUMN = 0

def calcStats(n, sum, sumsq):
    centroid = sum/n
    variance = (sumsq/n) - (pow((sum/n),2)) 
    std_dev = pow(variance,0.5)
    return centroid, variance, std_dev

def generateCompressedSets():
    global rs_set
    global cs_set
    global ds_set

    global ds_dict
    global cs_dict

    global no_of_clusters
    global increasedClusterSize
    global kLarge
    global removedColumnsIndices
    global noOfColumns

    clustersRemoved = list()
    decreasedKlarge = kLarge

    if(len(rs_set)<= increasedClusterSize):
        return

    if len(rs_set)<= decreasedKlarge:
        decreasedKlarge = int(0.8*len(rs_set))

    compressed_KMeans = KMeans(n_clusters=decreasedKlarge).fit(rs_set[:,2:])
    compressedClusterIndices = {1+label+no_of_clusters: np.where(compressed_KMeans.labels_ == label)[0] for label in range(compressed_KMeans.n_clusters)}
    
    for label, cluster in compressedClusterIndices.items():
        clusterLength =  len(cluster)

        if(clusterLength!=1):
            clustersRemoved.append(cluster)
            SUM = np.delete(rs_set[cluster], removedColumnsIndices, axis=ROW).sum(axis=COLUMN)
            SUMSQ = np.sum(np.square(np.delete(rs_set[cluster],removedColumnsIndices,axis=1).astype(np.float64)), axis = COLUMN)

            CENTROID, VAR, STD_DEV = calcStats(clusterLength, SUM, SUMSQ)

            cs_dict.update({label:(clusterLength, SUM, SUMSQ, CENTROID, VAR, STD_DEV)})
            rs_set[cluster,1] = label
            cs_set = cs_set + rs_set[cluster].tolist()
            
    if(clustersRemoved):
        rs_set = np.delete(rs_set, np.concatenate(clustersRemoved).reshape(-1*ROW), axis=COLUMN)

def calculateMD(centroid, std_dev, point):
    return np.sqrt(np.sum(np.square((point-centroid)/std_dev)))

def processChunk(chunk):
    global rs_set
    global cs_set
    global ds_set

    global ds_dict
    global cs_dict

    global threshold

    unassigned = list()

    for point in chunk:
        minSummary = [math.inf,""]
        actualPoint = point[2:]

        for label, clusterSummary in ds_dict.items():
            centroid = clusterSummary[CENTROID_INDEX]
            std_dev = clusterSummary[STD_DEV_INDEX]
            dist = calculateMD(centroid, std_dev, actualPoint)

            if dist< minSummary[0]:
                minSummary[0] = dist
                minSummary[1] = label
    
        if minSummary[0]< threshold:
            point[1] = minSummary[1]

            ds_summary = ds_dict[minSummary[1]]
            updatedN = ds_summary[N_INDEX] + 1
            updatedSum = ds_summary[SUM_INDEX] + actualPoint
            updatedSumSq = ds_summary[SUMSQ_INDEX] + np.square(actualPoint.astype(np.float64))

            updatedCentroid, updatedVariance, updatedStdDev = calcStats(updatedN, updatedSum, updatedSumSq)
            
            ds_dict.update({minSummary[1]: (updatedN, updatedSum, updatedSumSq, updatedCentroid, updatedVariance, updatedStdDev)})
            ds_set.append(point)
        
        else:
            unassigned.append(point)
    
    for point in unassigned:
        csMinSummary = [math.inf,""]
        csActualPoint = point[2:]

        for csLabel, csClusterSummary in cs_dict.items():
            csCentroid = csClusterSummary[CENTROID_INDEX]
            csStd_dev = csClusterSummary[STD_DEV_INDEX]
            csDist = calculateMD(csCentroid, csStd_dev, actualPoint)

            if csDist< csMinSummary[0]:
                csMinSummary[0] = csDist
                csMinSummary[1] = csLabel
    
        if csMinSummary[0]< threshold:
            point[1] = csMinSummary[1]

            cs_summary = cs_dict[csMinSummary[1]]
            csUpdatedN = cs_summary[N_INDEX] + 1
            csUpdatedSum = cs_summary[SUM_INDEX] + actualPoint
            csUpdatedSumSq = cs_summary[SUMSQ_INDEX] + np.square(actualPoint.astype(np.float64))

            csUpdatedCentroid, csUpdatedVariance, csUpdatedStdDev = calcStats(csUpdatedN, csUpdatedSum, csUpdatedSumSq)

            cs_dict.update({csMinSummary[1]: (csUpdatedN, csUpdatedSum, csUpdatedSumSq, csUpdatedCentroid, csUpdatedVariance, csUpdatedStdDev)})
            cs_set.append(point)
        
        else:
            point[1] = -1
            rs_set  = np.vstack([rs_set,point])
        
def mergeCS(newCS_dict, newCS_set):
    global rs_set
    global cs_set
    global cs_dict
    global threshold

    tempCS_set = np.array(cs_set)
    tempNewCS_set = np.array(newCS_set)

    for newLabel,newSummary in newCS_dict.items():
        minDist, minLabel = math.inf, ""
        newCentroid = newSummary[CENTROID_INDEX]

        for label, summary in cs_dict.items():
            centroid = summary[CENTROID_INDEX]
            std_dev = summary[STD_DEV_INDEX]
            currDist = calculateMD(centroid, std_dev, newCentroid)
            if currDist<minDist:
                minDist, minLabel = currDist, label

        if minDist<threshold:
            minSummary = cs_dict[minLabel]
            updatedN = minSummary[N_INDEX] + newSummary[0]
            updatedSum = minSummary[SUM_INDEX] + newSummary[1]
            updatedSumSq = minSummary[SUMSQ_INDEX] + newSummary[2]

            updatedCentroid, updatedVariance, updatedStdDev = calcStats(updatedN, updatedSum, updatedSumSq)

            addToCompressedSets = tempNewCS_set[tempNewCS_set[:,1]==newLabel]
            addToCompressedSets[:,1] = minLabel
            tempCS_set = np.vstack([tempCS_set,addToCompressedSets])
            tempNewCS_set = tempNewCS_set[tempNewCS_set[:,1]!=newLabel]

            cs_dict.update({minLabel:(updatedN, updatedSum, updatedSumSq, updatedCentroid, updatedVariance, updatedStdDev)})
        
        else:
            tempCS_set = np.vstack([tempCS_set, tempNewCS_set[tempNewCS_set[:,1]==newLabel]])
            cs_dict.update({newLabel:newSummary})
            tempNewCS_set = tempNewCS_set[tempNewCS_set[:,1]!=newLabel]
    
    for actualLabel, actualSummary in cs_dict.items():
        minDist, minLabel = math.inf, ""
        actualCentroid = actualSummary[3]

        for label, summary in cs_dict.items():
            if actualLabel!=label:
                centroid = summary[CENTROID_INDEX]
                std_dev = summary[STD_DEV_INDEX]
                currDist = calculateMD(centroid, std_dev, actualCentroid)

                if currDist < minDist:
                    minDist, minLabel = currDist, label

        if minDist < threshold:
            minSummary = cs_dict[minLabel]
            updatedN = minSummary[N_INDEX] + actualSummary[0]
            updatedSum = minSummary[SUM_INDEX] + actualSummary[1]
            updatedSumSq = minSummary[SUMSQ_INDEX] + actualSummary[2]

            updatedCentroid, updatedVariance, updatedStdDev = calcStats(updatedN, updatedSum, updatedSumSq)

            addToCompressedSets = tempCS_set[tempCS_set[:,1]==actualLabel]
            tempCS_set[np.where(tempCS_set[:,0]==actualLabel)] = minLabel

            cs_dict.update({minLabel:(updatedN, updatedSum, updatedSumSq, updatedCentroid, updatedVariance, updatedStdDev)})

    cs_set = tempCS_set.tolist()


def BFR(chunk):
    global rs_set
    global cs_set
    global ds_set

    global ds_dict
    global cs_dict

    global noOfColumns
    global increasedClusterSize
    global kLarge
    global removedColumnsIndices

    newCS_dict = {}
    newCS_set = []
    clustersRemoved = list()

    processChunk(chunk)

    # if(len(rs_set)<=increasedClusterSize):
    #     return

    decreasedKlarge = kLarge
    if (len(rs_set)<=decreasedKlarge):
        decreasedKlarge = int(0.8*len(rs_set))

    csKMeans = KMeans(n_clusters=decreasedKlarge).fit(rs_set[:,2:])

    rstart = max(cs_dict.keys())*2/2

    csTempClusterIndices = {rstart+1+label: np.where(csKMeans.labels_ == label)[0] for label in range(csKMeans.n_clusters)}

    for label, cluster in csTempClusterIndices.items():
        clusterLength = len(cluster)

        if clusterLength!=1:
            sum = np.delete(rs_set[cluster],removedColumnsIndices, axis = 1).sum(axis=0)
            sumsq = np.sum(np.square(np.delete(rs_set[cluster], removedColumnsIndices, axis = 1).astype(np.float64)), axis=0)

            centroid, var, std_dev = calcStats(clusterLength, sum, sumsq)

            newCS_dict.update({label: (clusterLength, sum, sumsq, centroid, var, std_dev)})
            rs_set[cluster,1] = label
            newCS_set = newCS_set + rs_set[cluster].tolist()
            clustersRemoved.append(cluster)

    if(clustersRemoved):
        rs_set = np.delete(rs_set, np.concatenate(clustersRemoved).reshape(-1), axis=0)

    mergeCS(newCS_dict, newCS_set)

def mergeCSDS():
    global rs_set
    global cs_set
    global ds_set

    global ds_dict
    global cs_dict

    tempCS_set = np.array(cs_set)
    tempDS_set = np.array(ds_set)

    clustersRemoved = list()

    for label, summary in cs_dict.items():
        minDist, minLabel = math.inf, ""
        centroid = summary[CENTROID_INDEX]

        for dsLabel, dsSummary in ds_dict.items():
            dsCentroid = dsSummary[CENTROID_INDEX]
            dsStdDev = dsSummary[STD_DEV_INDEX]
            currDist = calculateMD(dsCentroid,dsStdDev, centroid)
            if currDist<minDist:
                minDist, minLabel = currDist, dsLabel

        minSummary = ds_dict[minLabel]
        updatedN = minSummary[N_INDEX] + summary[N_INDEX]
        updatedSum = minSummary[SUM_INDEX] + summary[SUM_INDEX]
        updatedSumSq = minSummary[SUMSQ_INDEX] + summary[SUMSQ_INDEX]
        
        updatedCentroid, updatedVariance, updatedStdDev = calcStats(updatedN, updatedSum, updatedSumSq)

        addToDiscardSets = tempCS_set[tempCS_set[:,1] == label]
        addToDiscardSets[:,1] = minLabel
        tempDS_set = np.vstack([tempDS_set, addToDiscardSets])
        tempCS_set = tempCS_set[tempCS_set[:,1] != label]

        ds_dict.update({minLabel:(updatedN, updatedSum, updatedSumSq, updatedCentroid, updatedVariance, updatedStdDev)})

        clustersRemoved.append(label)

    for cluster in clustersRemoved:
        del cs_dict[cluster]

    cs_set = tempCS_set.tolist()
    ds_set = tempDS_set.tolist()


if __name__=='__main__':
    if len(sys.argv) !=4:
        print("All arguments haven't been specified")
        sys.exit(-1)

    start_time = time.perf_counter()
    
    input_file = sys.argv[1]
    no_of_clusters = int(sys.argv[2])
    output_file = sys.argv[3]

    cs_set = []
    ds_set = []

    ds_dict = {}
    cs_dict = {}

    input_data = np.loadtxt(input_file, delimiter=",")

    input_size = len(input_data)
    sample_size = int(input_size*SAMPLE_PERCENTAGE)

    init_data = input_data[:sample_size]

    _, noOfColumns = init_data.shape

    rs_set = np.array([]).reshape(0,noOfColumns)

    requiredDimesions = noOfColumns - 2
    threshold = 2*math.sqrt((noOfColumns - 2))

    increasedClusterSize = 10
    kLarge = increasedClusterSize * no_of_clusters
    skKMeans = KMeans(n_clusters=kLarge).fit(init_data[:,2:])

    tempClusterIndices = {label: np.where(skKMeans.labels_ == label)[0] for label in range(skKMeans.n_clusters)}

    for label, cluster in tempClusterIndices.items():
        if(len(cluster)<=increasedClusterSize):
            rs_set = np.vstack([rs_set,init_data[tempClusterIndices[label]]])
            rs_set[:,1] = -1
            init_data = np.delete(init_data, tempClusterIndices[label], axis = COLUMN)
    
    kMeansOriginal = KMeans(n_clusters=no_of_clusters).fit(init_data[:,2:])
    clusterIndcicesOriginal = {label: np.where(kMeansOriginal.labels_ == label)[0] for label in range(kMeansOriginal.n_clusters)}

    removedColumnsIndices = []
    removedColumnsIndices.append(0)
    removedColumnsIndices.append(1)

    for label, cluster in clusterIndcicesOriginal.items():
        N = len(cluster)
        SUM = np.delete(init_data[cluster],removedColumnsIndices, axis=ROW).sum(axis=COLUMN)
        SUMSQ = np.sum(np.square(np.delete(init_data[cluster],removedColumnsIndices,axis=ROW).astype(np.float64)), axis = COLUMN)

        CENTROID, VAR, STD_DEV = calcStats(N, SUM, SUMSQ)
       
        ds_dict.update({label:(N, SUM, SUMSQ, CENTROID, VAR, STD_DEV)})
        init_data[cluster,1] = label
        ds_set = ds_set + init_data[cluster].tolist()

    generateCompressedSets()

    round = 1
    intermediateResults = {}
    intermediateResults.update({round: (len(ds_set), len(cs_dict.keys()), len(cs_set), rs_set.shape[0])})
    round+=1

    start = int(input_size*SAMPLE_PERCENTAGE)
    end = start + int(input_size*SAMPLE_PERCENTAGE)

    while start<input_size:
        BFR(input_data[start:end])
        start = end
        end =  start + sample_size

        intermediateResults.update({round: (len(ds_set), len(cs_dict.keys()), len(cs_set), rs_set.shape[0])})
        round = round + 1

        if start>= input_size:
            mergeCSDS()

            clusters = np.vstack([ds_set,rs_set])
            clusters = clusters[clusters[:,0].argsort()]

            intermediateResults.update({int(round): (len(ds_set), len(cs_dict.keys()), len(cs_set), rs_set.shape[0])})

    clusterData = clusters[:,removedColumnsIndices].tolist()

    # with open(output_file,"w") as f:
    #     f.write("The intermediate results:\n")
    #     for key, val in intermediateResults.items():
    #         f.write("Round {}: {},{},{},{}".format(key,val[0],val[1],val[2],val[3]))
    #         f.write("\n")
    #     f.write("\n")

    #     f.write("The clusterting results:\n")
    #     for i in clusterData:
    #         f.write("{},{}".format(int(i[0]),int(i[1])))
    #         f.write("\n")

    with open(output_file,"w+",encoding="utf-8") as f:
        f.write("The intermediate results:")
        f.write("\n")
        f.write("\n".join('Round {}: {},{},{},{}'.format(key,val[0],val[1],val[2],val[3]) for key,val in intermediateResults.items()))
        f.write("\n\n")
        f.write("The clusterting results:")
        f.write("\n")
        f.write("\n".join('{},{}'.format(int(x[0]), int(x[1])) for x in clusters[:,[0,1]].tolist()))

    print(time.perf_counter()-start_time)

    from sklearn.metrics import normalized_mutual_info_score

    score = normalized_mutual_info_score(input_data[:, 1], clusters[:,1])

    print("Normalized Score: ", score)