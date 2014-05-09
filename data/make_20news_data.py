from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

def dump_sentences():
    corpus = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = corpus.data
    labels = corpus.target
    label_names = corpus.target_names
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
    preprocess = vectorizer.build_preprocessor()
    tokenize = vectorizer.build_tokenizer()
    
    def words(doc):
        p = preprocess(doc)
        return ' '.join(t.encode('ascii', 'replace') for t in tokenize(p))
    
    doccount = 0
    vocab = set()
    with open('20news.txt', 'w') as f:
        for doc, lbl in zip(docs, labels):
            w = words(doc)
            print >> f, label_names[lbl]
            print >> f, w
            doccount += 1
            vocab.update(w.split(' '))
    
    print 'Number of documents:', doccount
    print 'Number of unique words:', len(vocab)

if __name__ == '__main__':
    dump_sentences()

