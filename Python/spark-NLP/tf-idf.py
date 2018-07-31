"""
Coursework - Large-scale text classification
Task 1 - Read text files to create tf-idf vector
Aniuska Dominguez Ariosa
INM432 Big Data
MSc in Data Science (2014/15)
"""

#Remove Header of the content - it starts and ends with - *END*/***
#  Parameters
#  content: file's content
#
#  Return
#   tuple (idText, Content without header)
def removeHeader( content ):

    #Remove header and legal text as well to reduce file's content size
    #pattern = r'\*END\*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS\*Ver\.[0-9]{2}\.[0-9]{2}\.[0-9]{2}\*END\*(.*)'
    pattern = '.*?\*END.?\*(.*)'

    c = re.search(pattern,content,re.DOTALL)
    if c != None:
       c = c.group(1)
    else:
       c = content

    #Get file ID
    ## Id Text formats: [Etext #2043]/[EBook #12200]
    #idpos = re.search('#\d+\]',header)
    id = re.search('\[E(text|book) #(\d+)\]',content,re.IGNORECASE)
    idc = id.group(2)

    return (idc,c)
    
def remPlural(word): #remove the plural s at the end of word
   auxWord = word.lower()

   if len(auxWord) > 2 and auxWord.endswith('s') :
      auxWord = auxWord[:-1]

   return auxWord

def f_hashVector(f, wcl, vsize ):
    vec = [0] * vsize # initialise vector
    for (w,c) in wcl:
        i = hash(w) % vsize # get word index
        vec[i] = vec[i] + c # add tdidf to index
    return (f,vec)

if __name__ == "__main__": # if this is the program entrypoint
    if len(sys.argv) != 2: # check arguments
        print >> sys.stderr, "Usage: <script_name> <stopWords_filename> <path_Files> <path_Metafiles>"
        exit(-1) # exit if not enough arguments

    config = SparkConf().setMaster("local[2]")
    sc = SparkContext(conf=config, appName="Coursework - Large-scale text classification")
    
    #read stopWords from arg[1]
    stopwords = sc.textFile(sys.argv[1]).flatMap(lambda x: x.split(',')).collect()

    path = "/data/extra/gutenberg/text-tiny"

    for root, dirs, files in os.walk(path, topdown=False):
      for name in files:
          newFile = sc.wholeTextFiles( os.path.join(root, name))
          if "textFiles" in globals():
             textFiles = textFiles.union( newFile )
          else:
              textFiles = newFile

    fileNum = textFiles.count()
        
    #Remove files Header
    textIdFiles = textFiles.flatMap(lambda (f,x): [removeHeader(x)] )
    
    fileIdWords = textIdFiles.flatMap(lambda (f,x): [(f, w.strip()) for w in re.split('\W+',x)])
    fileIdWords = fileIdWords.filter(lambda (f,w): w not in stopwords and len(w) > 0)
    fileIdWords.cache()

    wordsCount = fileIdWords.map(lambda (f,x): ((f,remPlural(x)),1)).reduceByKey(add)
    wordsCount = wordsCount.filter(lambda x:  x[1] >= 5) # filter infrequent words

    wordsByFiles = wordsCount.map(lambda (fw,c): (fw[0],[(fw[1],c)]))
    wordsByFiles = wordsByFiles.reduceByKey(add)

    #Calculate IDF & TF 
    #create list of tuples (file,maxfrec,[(word,count)])
    wordsByFilesMaxFrecuency = wordsByFiles.map(lambda (f,wc) : ((f, max(wc, key = lambda l: l[1])), wc) )

    #Normalise - tf/maxfreq
    wordsByFilesTF = wordsByFilesMaxFrecuency.flatMap(lambda (fm,wc) : [([fm[0], w],[c,c*1.0/fm[1][1]]) for (w,c) in wc] )

    #create tuples (w,[file,tf])
    wordsTF = wordsByFilesTF.map(lambda (fw,cft): (fw[1],[(fw[0],cft[0],cft[1])] ) )
    wordsTF = wordsTF.reduceByKey(add)

    #add idf each tuple (w,[file,tf,idf]) - idf = log(N/df), df = len(list([]))
    wordsTFandIDF = wordsTF.map(lambda (w,fl): (w,[(f,ft,math.log10(fileNum/len(fl))) for (f,c,ft) in fl ]) )

    #calculate TF-IDF and create tuples (word,file,tfidf)
    wordsTFandIDF = wordsTF.map(lambda (w,fl): (w,[(f,ft,math.log10(fileNum/len(fl))) for (f,c,ft) in fl ]) )
    wordsTFIDF = wordsTFandIDF.flatMap(lambda (w,fl): [([w,f],ft*idf) for (f,ft,idf) in fl] )
    wordsTFIDF = wordsTFIDF.filter(lambda (wf,tfidf): tfidf != 0) #removing tfidf = 0 (irrelevant word) to reduce size

    #reajust tuples (f,[w,tfidf])
    fwl_tfidf = wordsTFIDF.map(lambda (fw,tfidf): (fw[1],[(fw[0],tfidf)] ) )
    fwl_tfidf = fwl_tfidf.reduceByKey(add)
    
    #create 10000 dimensional vector for doc using hashing
    #wfl : [word,file]; ftidf : wieght
    hashSize = 10
    fw_vec = fwl_tfidf.map(lambda (f,wcl): f_hashVector(f,wcl,hashSize) )