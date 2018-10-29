
# Large-scale text classification using Spark’s MapReduce with Python

## Why this code

- Big Data Technology
- Apply NLP techniques
- Use MapReduce programming model
- First Python coding


## How it works - background

- NLP for large-scale text classification ( > 100k books from project Gutenberg)
- Feature extraction using TF-IDF (Term frequency * Inverse Doc Frequency ). 
  TF ~ Document relevance & IDF ~ Term specificity
- MapReduce used for processing large volumes of data in parallel
- Spark is cluster-computing framework for general-purpose data processing engine (Big Data processing) 


## How it works - MapReduce explained

### Map
- Input is (key, value) tuples
- A custom function (business logic) is applied to every value in the tuple
- Output is a new list of (key,  value) tuples

### Reduce
- Tuples are sorted by key before to apply Reduce
- Input is (key, value) tuple from applying Map
- Custom function (usually aggregation or summation) to iterate the values for a given key. 
- Output is the final list of (key,  value) tuples


## How it works - Code explained

### Create Corpus


```python
    config = SparkConf().setMaster("local[2]")
    sc = SparkContext(conf=config, appName="Coursework - Large-scale text classification")

    #Create corpus
    #Read files content and get number of files in the corpus
    path = "/data/extra/gutenberg/text-tiny"
    for root, dirs, files in os.walk(path, topdown=False):
      for name in files:
          newFile = sc.wholeTextFiles( os.path.join(root, name))
          if "textFiles" in globals():
             textFiles = textFiles.union( newFile )
          else:
              textFiles = newFile

    fileNum = textFiles.count()  
    
    #read our list of commun & stop words
    stopwords = sc.textFile(sys.argv[1]).flatMap(lambda x: x.split(',')).collect()
```

### Text processing


```python
    #Text prepocessing
    #Create list of tuples (fileId, content) and remove file's header
    textIdFiles = textFiles.flatMap(lambda (f,x): [removeHeader(x)] )
    
    #Create list of tuple (file, list of words). Also remove stop words
    fileIdWords = textIdFiles.flatMap(lambda (f,x): [(f, w.strip()) for w in re.split('\W+',x)])
    fileIdWords = fileIdWords.filter(lambda (f,w): w not in stopwords and len(w) > 0)
    fileIdWords.cache()

    #Remove plural words (simple stemming) & word count
    #list of tuples ((fileID,word),1)
    wordsCount = fileIdWords.map(lambda (f,x): ((f,remPlural(x)),1)).reduceByKey(add)
    #tuples ((fileID,word),count)
    #filter infrequent words
    wordsCount = wordsCount.filter(lambda x:  x[1] >= 5)

    #Adjusting tuples (fileID, (word,count)) 
    wordsByFiles = wordsCount.map(lambda (fw,c): (fw[0],[(fw[1],c)]))
    #Grouping all (word,count) per FileID : tuples (filedID, [(word,count)])
    wordsByFiles = wordsByFiles.reduceByKey(add)
```

### Calculate TF.IDF


```python
    #Calculate Term frequency(TF) & Inverse Doc Frequency(IDF)
    #create list of tuples (fileID,maxfreq,[(word,count)])
    wordsByFilesMaxFrecuency = wordsByFiles.map(lambda (f,wc) : ((f, max(wc, key = lambda l: l[1])), wc) )

    #Normalise TF - tf/maxfreq (to discount for “longer” documents), creating list of tuples ([fileID,w],[count,TF])
    wordsByFilesTF = wordsByFilesMaxFrecuency.flatMap(lambda (fm,wc): [([fm[0], w],[c,c*1.0/fm[1][1]]) for (w,c) in wc])

    #Adjust tuples conveniently (word,[(fileID,count,TF )]) 
    wordsTF = wordsByFilesTF.map(lambda (fw,cft): (fw[1],[(fw[0],cft[0],cft[1])] ) )
    wordsTF = wordsTF.reduceByKey(add)
    
    #calculate TF-IDF and create new tuples (word,[(fileID,tf,idf)])
    #Adjust tuples & add idf where idf = log(N/df), df = len(list([]))
    wordsTFandIDF = wordsTF.map(lambda (w,fl): (w,[(f,ft,math.log10(fileNum/len(fl))) for (f,c,ft) in fl ]) )

    #Calculate TF-IDF, creating list of tuples ([word, fileID],tf-idf)
    wordsTFIDF = wordsTFandIDF.flatMap(lambda (w,fl): [([w,f],ft*idf) for (f,ft,idf) in fl] )
    #removing tfidf = 0 (irrelevant word) to reduce size
    wordsTFIDF = wordsTFIDF.filter(lambda (wf,tfidf): tfidf != 0) 

    #Reajust tuples (fileID,[word,tf-idf])
    fwl_tfidf = wordsTFIDF.map(lambda (fw,tfidf): (fw[1],[(fw[0],tfidf)] ) )
    fwl_tfidf = fwl_tfidf.reduceByKey(add)
    
    hashSize = 10
    fw_vec = fwl_tfidf.map(lambda (f,wcl): f_hashVector(f,wcl,hashSize) )
```
