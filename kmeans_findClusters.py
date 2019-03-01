#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import sys

# $example on$
from numpy import array
from math import sqrt
# $example off$

from pyspark import SparkContext
# $example on$
from pyspark.mllib.clustering import KMeans, KMeansModel
# $example off$

if __name__ == "__main__":
    sc = SparkContext(appName="KMeansExample")  # SparkContext

    def error(point):
        center = clusters.centers[clusters.predict(point)]
        return sqrt(sum([x**2 for x in (point - center)]))

    # $example on$
    # Load and parse the data
    data = sc.textFile(sys.argv[1])
    parsedData = data.map(lambda line: array([float(x) for x in line.split(',')]))

    WSSSEALL=[]
    k_range = range(1,13)
    for k in k_range:
   	 # Build the model (cluster the data)
	clusters = KMeans.train(parsedData, k , maxIterations=10, initializationMode="random")
	WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
	WSSSEALL.append("K= "+str(k)+" WSSE= "+str(WSSSE))

    print("Within Set Sum of Squared Error = " + str(WSSSEALL))



  
    # $example off$

    sc.stop()
