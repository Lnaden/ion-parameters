import numpy
import scipy
from sys import stdout
import pdb

'''
Implementation of DBSCAN clustering algorithm for use with regular spaced data and weights

This implementation accepts a weight matrix and a feature matrix of 3-dimensions and generates the neighborhoods based on the cluster they belong to. 

Returns a 3-dimension matrix of the same shape as the weight and feature matrix with a new feature block based on neighborhoods
'''

class dbscan(object):
    def __init__(self, feature_block, weights, tolerance=0.1, min_points=5):
        self.feature = feature_block
        self.weights = weights
        self.dims = self.feature.shape
        self.ndims = len(self.dims)
        self.tol = tolerance
        self.min_points = min_points
        if min_points < 2:
            print "Warrning: Having fewer than 2 points for minimum cluster size will make each point its own cluster, use at own risk!"
        return
    def generate_neighborhoods(self):
        #This is a slow process since it acts on each index
        #Lay out neighborhoods
        self.neighborhoods = numpy.zeros(self.dims, dtype=numpy.int32)
        self.nc = 0 #Number of clusters
        #Lay out visitation
        self.visitation = numpy.zeros(self.dims, dtype=bool)
        self.visitation[self.feature == 0] = True
        #Loop through each point
        #Set up iterator to grab values and index (mostly index)
        it = numpy.nditer(self.feature, flags=['multi_index'])
        counter=1
        while not it.finished:
            index = numpy.array(it.multi_index)
            stdout.write('\r{0:d}/{1:d}'.format(counter,self.feature.size))
            stdout.flush()
            if not self.visitation[tuple(index)]: #Go over unvisited point
                self.visitation[tuple(index)] = True
                neighbors = self.regionQuery(index)
                if neighbors.shape[0] >= self.min_points:
                    #Add P to new cluster C
                    self.nc +=1
                    self.neighborhoods[tuple(index)] = self.nc
                    self.expandClusters(index, neighbors, self.nc)
            counter+=1
            it.iternext() #DO NOT FORGET THIS LINE
        stdout.write('\n')
        return self.neighborhoods, self.nc 
    def relErr(self, true, aprox):
        return numpy.abs((true-aprox)/true)
    def regionQuery(self, index):
        #Create the skeleton region, cluster class works for this too
        pivot = index
        region = clusterset(pivot)
        #Cycle through each index +/-
        for dx in [-1,0,1]:
            if pivot[0]+dx >= 0 and pivot[0]+dx < self.dims[0]: #Check overbounds
                for dy in [-1,0,1]:
                    if pivot[1]+dy >= 0 and pivot[1]+dy < self.dims[1]: #Check overbounds
                        for dz in [-1,0,1]:
                            if pivot[2]+dz >= 0 and pivot[2]+dz < self.dims[2]: #Check overbounds
                                #Check if point is within relative error
                                pshift = numpy.array([dx,dy,dz])
                                pndx = pivot + pshift
                                Wij = self.weights[tuple(pivot)]
                                Pij = self.weights[tuple(pndx)]
                                #Ensure its within high err region, and "close" in error
                                if self.feature[tuple(pndx)] > 0 and (self.relErr(Wij, Pij) <= self.tol or self.relErr(Pij, Wij) <= self.tol) and not numpy.array_equal(pndx, pivot):
                                    #Omited self to prevent overcounting cluster
                                    region.add(pndx)
        return region.points
    def expandClusters(self, index, neighbors, clusterid):
        #loop through each neighbor
        Nneighbors = neighbors.shape[0]
        i = 0 
        while i < neighbors.shape[0]:
            neighbor = neighbors[i,:]
            if not self.visitation[tuple(neighbor)]:
                self.visitation[tuple(neighbor)] = True
                newneighbors = self.regionQuery(neighbor)
                if newneighbors.shape[0] >= self.min_points:
                    neighbors = numpy.concatenate((neighbors, newneighbors))
            if self.neighborhoods[tuple(neighbor)] == 0:
                self.neighborhoods[tuple(neighbor)] = clusterid
            i +=1 #Don't forget me!
        return            

class clusterset(object):
    '''
    House a set of indicies for points
    '''
    def __init__(self,point):
        self.__basepoint = numpy.zeros([1,3],dtype=numpy.int32)
        self.points = self.__basepoint.copy()
        self.points[0,:] = point
    def add(self,points):
        if len(points.shape) == 1:
            points = numpy.array([points])
        #self.points = numpy.append(self.points, self.__basepoint)
            self.points = numpy.concatenate((self.points, points))

if __name__=="__main__":
    print "Syntax Good, boss" 
