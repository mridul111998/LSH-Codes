import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from nltk import word_tokenize
from nltk.corpus import reuters 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")

def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text));
    words = [word for word in words if word not in cachedStopWords]
    tokens =(list(map(lambda token: PorterStemmer().stem(token), words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token)>=min_length, tokens));
    return filtered_tokens

# Return the representer, without transforming
def tf_idf(docs):
    
    tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=3, max_df=0.90, max_features=1000, use_idf=True, sublinear_tf=True);
    tfidf.fit(docs);
    return tfidf;

def feature_values(doc, representer):
    doc_representation = representer.transform([doc])
    features = representer.get_feature_names()
    return [(features[index], doc_representation[0, index]) for index in doc_representation.nonzero()[1]]

def collection_stats():
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents");
	
    train_docs = list(filter(lambda doc: doc.startswith("train"), documents));
    print(str(len(train_docs)) + " total train documents");
	
    test_docs = list(filter(lambda doc: doc.startswith("test"), documents));	
    print(str(len(test_docs)) + " total test documents");

    # List of categories 
    categories = reuters.categories();
    print(str(len(categories)) + " categories");

    # Documents in a category
    category_docs = reuters.fileids("acq");

    # Words for a document
    document_id = category_docs[0]
    document_words = reuters.words(category_docs[0]);
    print(document_words);	

    # Raw document
    print(reuters.raw(document_id));

def main():
    train_docs = []
    test_docs = []
    print 'mridul'
    #print reuters.fileids()
    # testing for first 2 documents
    vocab = set()
    documents = []
    size = 2
    for doc_id in reuters.fileids()[:size]:
#        if(doc_id.startswith("train")):
 #           train_docs.append(reuters.raw(doc_id))
  #      else:
   #         test_docs.append(reuters.raw(doc_id))
#       print tokenize(reuters.raw(doc_id))
        fq= defaultdict( int )
        for i in  tokenize(reuters.raw(doc_id)):
            #going through tokenized document
            fq[i] += 1
        # fq contains the word count of a document
        print len(fq)
        documents.append(fq)
        for i in fq.keys():
            vocab.add(i)
    print 'vocab = ',len(vocab)
    vocab = list(vocab)
    counter = -1
    coordinates = np.zeros(shape=(size,len(vocab)))
    print coordinates.shape
    print len(documents)

    for i in documents:
        print 'sharma'
        counter = counter + 1
        coordinate = [0]*len(vocab)
        for word in i.keys():
          #  print i[word]
            coordinate[vocab.index(word)] = i[word]
        coordinates[counter] = coordinate
    print coordinates.shape
    print coordinates
main()
