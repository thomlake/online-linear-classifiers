import urllib, tarfile, os, re
from sklearn.feature_extraction.text import CountVectorizer

URL = 'http://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz'
BASEDIR = './sorted_data_acl/'
TOPICS = ('books', 'dvd', 'electronics', 'kitchen_&_housewares')
DIRS = [os.path.join(BASEDIR, topic) for topic in TOPICS]
POSREV = 'positive.review'
NEGREV = 'negative.review'
REVREGEX = re.compile(r'(?<=<review_text>)([\s\S]*?)(?=</review_text>)')

def download():
    if not os.path.exists('./reviews.gz'):
        print 'downloading reviews'
        urllib.urlretrieve(URL, 'reviews.gz')
    if not os.path.exists(BASEDIR):
        print 'extracting reviews'
        tfile = tarfile.open('reviews.gz', 'r:gz')
        tfile.extractall('.')

def dump_reviews():
    download()
    print 'making dataset'
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
    preprocess = vectorizer.build_preprocessor()
    tokenize = vectorizer.build_tokenizer()
    
    def dumbascii(thing):
        try:
            thing.encode('ascii', 'replace')
            return True
        except UnicodeDecodeError:
            return False

    def words(doc):
        p = preprocess(doc)
        return ' '.join(t.encode('ascii', 'replace') for t in tokenize(p) if dumbascii(t))
    
    doccount = 0
    vocab = set()
   
    with open('reviews.txt', 'w') as fout:
        for topicdir in DIRS:
            with open(os.path.join(topicdir, POSREV), 'r') as f:
                text = f.read()
            for doc in REVREGEX.findall(text):
                w = words(doc)
                print >> fout, 'positive'
                print >> fout, w
                doccount += 1
                vocab.update(w.split(' '))

            with open(os.path.join(topicdir, NEGREV), 'r') as f:
                text = f.read()
            for doc in REVREGEX.findall(text):
                w = words(doc)
                print >> fout, 'negative'
                print >> fout, w
                doccount += 1
                vocab.update(w.split(' '))
    print 'Number of documents:', doccount
    print 'Number of unique words:', len(vocab)

if __name__ == '__main__':
    dump_reviews()
