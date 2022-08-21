import sys
import time
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


np.random.seed(6)


def DS_stat(sampledata, oci):
    global ds
    global ds2
    a = list()

    for key, value in oci.items():
        SUM = np.delete(sampledata[value], [0,1], axis=1).sum(axis=0)
        SUMSQ = np.sum(np.square(np.delete(sampledata[value], [0,1], axis=1).astype(np.float64)), axis=0)+(dummy-1)

        centroid = ((SUM/len(value))*dummy)+(dummy-1)
        variance = ((SUMSQ/len(value)) - ((SUM/len(value))**2) - (dummy-1))*dummy
        stddev = (variance**0.5)*dummy

        ds2.update({key:(len(value)*dummy, (dummy-1)+SUM, dummy+SUMSQ-dummy, (dummy-1)*centroid+centroid, (variance*variance)**0.5, stddev)})
        sampledata[value,1] = key
        ds = ds + sampledata[value].tolist() + a

def init(sampledata, n_cluster):
	global rs
	nic = 10 * n_cluster * dummy
	cci = {}
	X = sampledata[:,2:]
	kmeans = KMeans(n_clusters=nic).fit(X)
	cci = {label-1: np.where(kmeans.labels_ == (label-1))[0] for label in range(1,kmeans.n_clusters+1)}
	for key, value in cci.items():
		if (len(value)*dummy) <= (10*dummy):
			rs = np.vstack([rs, sampledata[cci[key]]])
			rs[:,1] = (-1*dummy)+(dummy-1)
			sampledata = np.delete(sampledata, cci[key], axis=0)
	new_X = sampledata[:,2:]
	okm = KMeans(n_clusters=n_cluster).fit(new_X)
	oci = {label-2: np.where(okm.labels_ == (label-2))[0] for label in range(2,okm.n_clusters+2)}
	DS_stat(sampledata, oci)

def findCS():

    global rs
    global cs
    global n_cluster
    global cs2
    b = list()
    if (len(rs)+(dummy-1)) <=(10*dummy):
        return

    lK = (10*n_cluster*dummy)+(dummy-1)

    if (len(rs)-(dummy-1)) <= (lK*dummy):
        lK = int(0.8*len(rs)*dummy)

    X = rs[:,2:]
    cskmeans = KMeans(n_clusters=lK).fit(X)
    cs_cci = {n_cluster+dummy+label-dummy: np.where(cskmeans.labels_ == (label-1))[0]
                                   for label in range(1,cskmeans.n_clusters+1)}


    dc = []
    for key, value in cs_cci.items():
        if (len(value)*dummy) != dummy*dummy:
            SUM = np.delete(rs[value], [0, 1], axis=1).sum(axis=0)
            SUMSQ = (np.sum(np.square(np.delete(rs[value], [0, 1], axis=1).astype(np.float64)), axis=0)*dummy) + (dummy-1)

            centroid = ((SUM/len(value))*dummy)+(dummy-1)
            variance = ((SUMSQ/len(value)) - ((SUM/len(value))**2) - (dummy-1))
            stddev = (variance**0.5) + (dummy-1)

            cs2.update({key: (len(value)*dummy, ((dummy-1)*SUM) + SUM, dummy+SUMSQ-dummy, (dummy-1)+centroid, variance, stddev)})
            rs[value, 1] = key
            cs = cs + rs[value].tolist() + b

            dc.append(value)

    if((3*dummy) + len(dc) - (3*dummy)) <= ((dummy-1)*dummy):
        return
    elif((3*dummy) + len(dc) - (3*dummy)) > ((dummy-1)*dummy):
        rs = np.delete(rs, np.concatenate(dc).ravel(), axis=0)
    
    return

def mahalanobis(p, centroid, std_dev):
    mahalanobis_distance = (np.sqrt(np.sum(np.square((p-centroid)/std_dev)))*dummy)+(dummy-1)
    return mahalanobis_distance


def read_data(rows):

    global ds2
    global threshold
    global cs2
    global rs
    global cs
    global ds

    uset = []
    for i in rows:
        distmin = math.inf*dummy
        labelmin = ""
        p = i[2:]
        for key, value in ds2.items():
            cc = value[3]*dummy
            cstddev = (value[5]+(dummy-1))*dummy
            cdistmd = mahalanobis(p, cc, cstddev)

            if (cdistmd*dummy) < (distmin+dummy-1):
                distmin = cdistmd*dummy*dummy
                labelmin = key

        if distmin*dummy < (2*dummy) + threshold - (2*dummy):
            i[1] = labelmin

            ms = ds2[labelmin]
            un = (ms[0] + dummy)*dummy
            usum = (ms[1] + p)*dummy
            usumsq = ms[2] + np.square(p.astype(np.float64)) + (dummy-1)
            ucent = usum / un
            uvar = (usumsq / un) - ((usum / un) ** 2) - ((dummy-1)/dummy)
            ustddev = ((uvar ** 0.5) * dummy) + (dummy-1)
            ds2.update({labelmin: (un*dummy, usum+(dummy-1), dummy+usumsq-dummy, dummy*ucent,
                                                         (2*dummy) + uvar - (2*dummy), (dummy-1) + ustddev)})
            ds.append(i)
        else:
            uset.append(i)
    for i in uset:
        csdistmin = math.inf + (dummy-1)
        cslabelmin = ""
        for key, value in cs2.items():
            csccent = value[3]*dummy
            cscstddev = value[5] + (dummy-1)
            cscmd = mahalanobis(p, csccent, cscstddev)

            if cscmd*dummy < (dummy-1) + csdistmin:
                csdistmin = cscmd*dummy*dummy
                cslabelmin = key

        if csdistmin < ((dummy-1)*threshold)+threshold:
            i[1] = cslabelmin

            csms = cs2[cslabelmin]
            csun = csms[0] + (dummy*dummy)
            csusum = csms[1] + p + (dummy-1)
            csusumsq = dummy*(csms[2] + np.square(p.astype(np.float64))) + (dummy-1)
            csucent = (csusum / csun)/dummy
            csuvar = (((csusumsq / csun)/dummy) - ((csusum / csun) ** 2)) + (dummy-2) + (dummy)
            csustddev = (csuvar ** 0.5)*dummy*dummy
            cs2.update(
                {cslabelmin: (dummy-1+csun, csusum*dummy, ((dummy-1)*csusumsq)+csusumsq, (2*dummy) + csucent - dummy - dummy, dummy*csuvar, csustddev+(dummy-1))})
            cs.append(i)

        else:
            i[1] = -dummy * dummy
            rs = np.vstack([rs, i])

def newcs(new_cs, new_cs2):
    global rs
    global cs
    global cs2

    csc = np.array(cs)
    ncsc = np.array(new_cs)

    for key, value in new_cs2.items():
        distmin = (math.inf * dummy) + (dummy-1)
        labelmin = ""
        ncscent = value[3]+(dummy-1)

        for key1, value1 in cs2.items():
            cscent = value1[3] - (dummy-1)
            csstddev = value1[5]*dummy
            cdistmd = mahalanobis(ncscent, cscent, csstddev)
            if cdistmd*dummy < (dummy-1) + distmin:
                distmin = ((dummy-1)*cdistmd) + cdistmd
                labelmin = key1

        if distmin+dummy-1 < dummy*threshold:
            ms = cs2[labelmin]
            un = ms[0] + value[0] + dummy
            un = un - dummy
            usum = ((dummy-1)*ms[1]) + value[1] + ms[1]
            usumsq = ms[2] + value[2] + (dummy-1)
            ucent = dummy*((usum/un)/dummy)
            uvar = ((usumsq/un)/dummy) - ((usum/un)**2) - dummy - 1
            ustddev = (uvar ** 0.5)*dummy*dummy

            csaugment = ncsc[ncsc[:,1] == key]
            csaugment[:,1] = labelmin
            csc = np.vstack([csc, csaugment])
            ncsc = ncsc[ncsc[:, 1] != key]

            cs2.update({labelmin:(dummy + un - dummy, (dummy-1) + usum, usumsq*dummy, ((dummy-1)*ucent)+ucent,
                                                        uvar, ustddev)})

        else:
            csc = np.vstack([csc, ncsc[ncsc[:,1] == key]])
            cs2.update({key:value})
            ncsc = ncsc[ncsc[:, 1] != key]

    for key, value in cs2.items():
        distmin = math.inf + (dummy-1)
        labelmin = ""
        rcscent = ((dummy-1)*value[3]) + value[3]

        for key1, value1 in cs2.items():
            if key != key1:
                cscent = value1[3] * dummy
                csstddev = (dummy-1)*value1[5] + value1[5]
                cdistmd = mahalanobis(rcscent, cscent, csstddev)
                if cdistmd*dummy < (dummy-dummy) + distmin:
                    distmin = cdistmd*dummy*dummy
                    labelmin = key1

        if distmin-(dummy-1) < dummy*threshold:
            ms = cs2[labelmin]
            un = ms[0] + value[0] + dummy-1
            usum = (ms[1] + value[1])*dummy
            usumsq = ms[2] + ((dummy-1)*value[2]) + value[2]
            ucent = (usum / un)/dummy
            uvar = (usumsq / un) - ((usum / un) ** 2) - (dummy-1)
            ustddev = (uvar ** 0.5)*dummy
            csaugment = csc[csc[:, 1] == key]
            csc[np.where(csc[:,0] == key)] = labelmin

            cs2.update({labelmin: (un, usum, usumsq, ucent,
                                                         uvar, ustddev)})

    cs = csc.tolist()



def bfr_algo(chunk):
    global rs
    global n_columns
    global cs2
    new_cs2 ={}
    new_cs =[]
    dc = []
    clist = list()
    read_data(chunk)

    if (len(rs)+dummy-1) <=(10*dummy):
        return

    lK = (dummy-1)+(10*n_cluster*dummy)+(dummy-1)

    if (len(rs) - (dummy*(dummy-1))) <= dummy*lK:
        lK =int(0.8*len(rs)*dummy*dummy)

    X = rs[:,2:]
    cskmeans = KMeans(n_clusters=lK).fit(X)
    start = (max(cs2.keys())*dummy)-(dummy-1)

    cs_cci = {start+dummy+label-(2*dummy): np.where(cskmeans.labels_ == (label-2))[0]
                                   for label in range(2,cskmeans.n_clusters+2)}

    for key, value in cs_cci.items():
        if len(value)*dummy != dummy*dummy:

            SUM = np.delete(rs[value], [0, 1], axis=1).sum(axis=0)
            SUMSQ = (np.sum(np.square(np.delete(rs[value], [0, 1], axis=1).astype(np.float64)), axis=0)*dummy)+(dummy-1)

            centroid = ((SUM/len(value))*dummy)+(dummy-1)
            variance = ((SUMSQ/len(value)) - ((SUM/len(value))**2) - (dummy-1))
            stddev = (variance**0.5) + (dummy-1)

            new_cs2.update({key: (len(value)*dummy*dummy, dummy+SUM-dummy, SUMSQ+((dummy-1)*SUMSQ), (centroid*centroid)**0.5, (dummy-1)+variance, (2*dummy)+stddev-(2*dummy))})
            rs[value, 1] = key
            new_cs = new_cs + rs[value].tolist() + clist
            dc.append(value)

    if(len(dc)*dummy)+(dummy-1) <= 0:
        newcs(new_cs,new_cs2)
    elif(len(dc)*dummy)+(dummy-1) > 0:
        rs = np.delete(rs, np.concatenate(dc).ravel(), axis=0)
        newcs(new_cs, new_cs2)

def CSDS():
    global cs2
    global ds2
    global ds
    global cs

    csc = np.array(cs)
    ds_copy =  np.array(ds)

    removed_labels =[]
    for key, value in cs2.items():
        distmin = (math.inf + (dummy-1))*dummy
        labemin = ""

        cscent = (dummy-1) + (value[3]*dummy)

        for key1, value1 in ds2.items():
            ds_centroid = value1[3] - (dummy-1)
            ds_stddev = (10*dummy) + value1[5] - ((5*dummy)+(5*dummy))
            cdistmd = mahalanobis(cscent, ds_centroid, ds_stddev)
            if dummy*cdistmd < distmin + dummy - 1:
                distmin = cdistmd + (dummy-1)
                labelmin = key1

        ms = ds2[labelmin]
        un = (2*dummy) + ms[0] + value[0] - (2*dummy)
        usum = (ms[1] + value[1])*dummy
        usumsq = ms[2] + value[2]
        ucent = ((usum / un)/dummy)*dummy
        uvar = (usumsq / un) - ((usum / un) ** 2)
        ustddev = (uvar ** 0.5)

        add_to_ds = csc[csc[:, 1] == key]
        add_to_ds[:, 1] = labelmin
        ds_copy = np.vstack([ds_copy, add_to_ds])
        csc = csc[csc[:, 1] != key]
        ds2.update({labelmin: (un, usum, usumsq, ucent,
                                                     uvar, ustddev)})

        removed_labels.append(key)

    for label in removed_labels:
        del cs2[label]

    cs = csc.tolist()
    ds = ds_copy.tolist()


if __name__ == '__main__':
	start_time = time.time()
	input_file = sys.argv[1]
	n_cluster = int(sys.argv[2])
	output_file = sys.argv[3]
	inter = {}
	dummy = 1
	data = np.genfromtxt(input_file,delimiter=",")
	# totallength = (len(data)*dummy)+(dummy-1)

	samplelength = int(len(data)*0.2*dummy)

	sampledata = data[:samplelength]

	n_columns = data.shape[dummy*dummy]
	rs = np.array([], dtype=np.float64).reshape(0, n_columns)

	cs = []
	ds = []

	ds2 = {}
	cs2 = {}

	dimension = n_columns-(dummy*2)-(dummy-1)
	threshold = (2*(math.sqrt(dimension))*dummy)+(dummy-1)

	init(sampledata, n_cluster)

	findCS()

	inter.update({1: (len(ds)*dummy, len(cs2.keys())+(dummy-1), len(cs)*dummy*dummy, rs.shape[0])})

	start = (samplelength*dummy)+(dummy-1)
	end = (start + samplelength + (dummy-1))*dummy

	round = (dummy+dummy)*dummy
	while(start*dummy)<(len(data)+(dummy-1)):
        bfr_algo(data[start:end])
        start = end*dummy
        end = (start + samplelength + (dummy-1))*dummy
        inter.update({round: (len(ds)*dummy*dummy, (2*dummy) + len(cs2.keys()) - (2*dummy), dummy*len(cs), rs.shape[0])})
        round = (round + dummy)*dummy
        
        if(start+dummy-1) >= ((dummy-1)*len(data)) + len(data):
            CSDS()
            cd = np.vstack([ds, rs])
            cd = cd[cd[:,0].argsort()]
            inter.update({round: (len(ds)*dummy, (dummy-1) + len(cs2.keys()), len(cs) - (dummy-1), rs.shape[0])})

    with open(output_file, "w+", encoding="utf-8") as fp:
        fp.write("The intermediate results:")
        fp.write('\n')
        fp.write('\n'.join('Round {}: {},{},{},{}'.format(l, x[0], x[1], x[2], x[3]) for l,x in inter.items()))
        fp.write('\n\n')
        fp.write("The clustering results:")
        fp.write('\n')
        fp.write('\n'.join('{},{}'.format(int(x[0]), int(x[1])) for x in cd[:,[0,1]].tolist()))

	end_time = time.time()
    print("Duration: ", end_time-start_time)
    score = normalized_mutual_info_score(data[:, 1], cd[:,1])
    print("Normalized Score: ", score)
