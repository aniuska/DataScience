"""
Coursework - Large-scale text classification
Task 2 - Read metada files to get subjects er file
Aniuska Dominguez Ariosa
INM432 Big Data
MSc in Data Science (2014/15)
"""

import sys
import re
import os
import math
import pickle

from pyspark import SparkContext
from pyspark.conf import SparkConf
from operator import add


import xml.etree.ElementTree as ET
if __name__ == "__main__": # if this is the program entrypoint
    if len(sys.argv) != 3: # check arguments
        print >> sys.stderr, "Usage: <script_name> <path_Metafiles_to_read><pickle_file_to save_results>"
        exit(-1) # exit if not enough arguments

    config = SparkConf().setMaster("local[2]")
    sc = SparkContext(conf=config, appName="Parsing XML files subjects: Coursework - Large-scale text classification")

    #Read Metadata from XML files
    #metapath = "meta"
    #metapath = '/data/extra/gutenberg/meta'
    metapath = sys.argv[1]
    
    subjectsList = []

    print("\n************* Reading metadata files ........")
    #read meta files
    for root, dirs, files in os.walk(metapath, topdown=False):
     for name in files:
      tree = ET.parse(os.path.join(root, name))
      rootXML = tree.getroot()
      
      #get ebook id
      ebook = rootXML.find('{http://www.gutenberg.org/2009/pgterms/}ebook')
      ebookId = ebook.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
      pos = ebookId.find('/')
      if pos != -1:
        ebookId = ebookId[pos+1:-1]

      #print ("id",ebookId)

      #reading subjects
      for child in ebook:
          #print "\ntag: %s, attr: %s" % (child.tag, child.attrib)
          if child.tag == '{http://purl.org/dc/terms/}subject':
              grandchildren = list(child.iter())
              for ele in grandchildren:
                if ele.tag == '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}value':
                  sublist = re.split(',|-{2}', ele.text)
                  #append tuple (id,sublist) in subjetsList
                  #verify list is not empty
                  #for s in sublist: s.strip('')
                  subjectsList.append((ebookId, sublist))

    print("\n************* RDD convertion ........")
    #Convert list to RDD and save it to use later
    rdd = sc.parallelize(subjectsList)
    rdd = rdd.glom()
    #rdd = rdd.flatMap(lambda (f,sl): (f, [s.strip() for s in sl]) ).reduceByKey(add)
    rdd = rdd.flatMap(lambda (fl): fl).reduceByKey(add)

    print "############################################################\n"
    print "\nTen first subjects of %d\n" % rdd.count()
    output = rdd.take(10)
    for index in output:
        print("*** ",index)

    #Save subject list to file to use later
    #rdd.saveAsPickleFile('sl')

    print("\n************* Saving pkl files ........")
    sub_list = rdd.take(2)
    filename = sys.argv[2]
    if not filename.endswith(".pkl"):
        filename += ".pkl"

    rdd.saveAsPickleFile(filename)

    """
    f = open(filename,'w')
    pickle.dump(sub_list,f)
    f.close()
    print "\nThe file %s has been created" % (filename)
    sc.stop()
    """



