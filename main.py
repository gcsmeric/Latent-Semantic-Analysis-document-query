from gensim.models import KeyedVectors
import numpy as np
from sklearn.utils.extmath import randomized_svd
from math import *
import re
import collections

def top_k_unigrams(tweets, stop_words, k):
    unidict = {}
    pattern = re.compile(r"[a-z#].*")
    for tw in tweets:
        t = tw.split()
        for w in t:
            if (w not in stop_words) and pattern.match(w):
                if w not in unidict:
                    unidict[w] = 1
                else:
                    unidict[w] += 1
    if k==-1:
        return unidict
    
    top_k_unigrams = sorted(unidict, key=unidict.get, reverse=True)[:k]
    top_k_dict = {}
    for unigram in top_k_unigrams:
        top_k_dict[unigram] = unidict[unigram]
    return top_k_dict

def context_word_frequencies(tweets, stop_words, context_size, frequent_unigrams):
    context_counter = {}
    for tw in tweets:
        t=tw.split()
        for i in range(len(t)):
            w=t[i]
            for j in range(context_size):
                if i-(j+1)>=0 and t[i-(j+1)] not in stop_words and (t[i-(j+1)][0].isalpha() or t[i-(j+1)][0]=="#") and t[i-(j+1)] in frequent_unigrams:
                    if (w, t[i-(j+1)]) not in context_counter:
                        context_counter[(w, t[i-(j+1)])] = 1
                    else:
                        context_counter[(w, t[i-(j+1)])] += 1
                if i+(j+1)<len(t) and t[i+(j+1)] not in stop_words and (t[i+(j+1)][0].isalpha() or t[i+(j+1)][0]=="#") and t[i+(j+1)] in frequent_unigrams:
                    if (w, t[i+(j+1)]) not in context_counter:
                        context_counter[(w, t[i+(j+1)])] = 1
                    else:
                        context_counter[(w, t[i+(j+1)])] += 1

    return context_counter

def pmi(word1, word2, unigram_counter, context_counter):
    n=0
    for v in unigram_counter.values():
        n+=v
    if (word1, word2) in context_counter:
        cooqfreq = context_counter[(word1, word2)]
    else:
        cooqfreq = 1
    if word1 in unigram_counter:
        w1freq = unigram_counter[word1]
    else:
        w1freq = 1
    if word2 in unigram_counter:
        w2freq = unigram_counter[word2]
    else:
        w2freq = 1
    pmi = log2((cooqfreq*n)/(w1freq*w2freq))
    return pmi

def build_word_vector(word1, frequent_unigrams, unigram_counter, context_counter):
    vector = {}
    for w in frequent_unigrams:
        if (word1, w) not in context_counter:
            vector[w] = 0.0
        else:
            vector[w] = pmi(word1, w, unigram_counter, context_counter)
    return vector

def get_top_k_dimensions(word1_vector, k):
    topd = {}
    topk = sorted(word1_vector, key=word1_vector.get, reverse=True)[:k]
    for w in topk:
        topd[w] = word1_vector[w]

    return topd

def get_cosine_similarity(word1_vector, word2_vector):
    numerator = 0
    denom1 = 0
    denom2 = 0
    for w in word1_vector:
        numerator += word1_vector[w]*word2_vector[w]
        denom1 += pow(word1_vector[w],2)
        denom2 += pow(word2_vector[w], 2)

    cos_sim = numerator/(sqrt(denom1)*sqrt(denom2))
    return cos_sim

def get_most_similar(word2vec, word, k):
    if not word in word2vec.key_to_index:
        return []
    result = word2vec.most_similar(word, topn=k)
    return result

def word_analogy(word2vec, word1, word2, word3):
    result = word2vec.most_similar(positive=[word3, word2], negative=[word1])
    return result[0]

def create_tfidf_matrix(documents, stopwords):
    documents_processed = []
    vocab_set = set()
    for i in range(len(documents)):
        documents_processed.append([])
        for w in documents[i]:
            wlow = w.lower()
            if not wlow in stopwords and w.isalnum():
                documents_processed[i].append(wlow)
                vocab_set.add(wlow)
    res = np.zeros(shape=(len(documents),len(vocab_set)))
    vocab = sorted(list(vocab_set))
    
    #counting words per doc
    unigram_counts = [{} for _ in range(len(documents))]
    for i in range(len(unigram_counts)):
        for j in range(len(vocab)):
            unigram_counts[i][vocab[j]] = 0

    for i in range(len(documents_processed)):
        for w in documents_processed[i]:
            unigram_counts[i][w]+=1
    #finding portion of docs containing each word
    word_contained = []
    for i in range(len(vocab)):
        word_contained.append(0)
        for j in range(len(documents)):
            if unigram_counts[j][vocab[i]]>0:
                word_contained[i]+=1

    for i in range(len(documents)):
        print(i)
        #doc counting
        #reimplement
        for j in range(len(vocab)):
            res[i,j]=(unigram_counts[i][vocab[j]])*(np.log10(len(documents)/(word_contained[j]+1))+1)
            #(i,j) cell should have tf-idf score of ith doc and jth word in vocab

    #I was having a similar issue. Make sure youre using the formula from the wikipedia page for idf: idf = np.log10((N) / (n + 1)) + 1
    return res, vocab

def get_idf_values(documents, stopwords):
    documents_processed = []
    vocab_set = set()
    for i in range(len(documents)):
        documents_processed.append([])
        for w in documents[i]:
            wlow = w.lower()
            if not wlow in stopwords and w.isalnum():
                documents_processed[i].append(wlow)
                vocab_set.add(wlow)
    res = np.zeros(shape=(len(documents),len(vocab_set)))
    vocab = sorted(list(vocab_set))
    
    #counting words per doc
    unigram_counts = [{} for _ in range(len(documents))]
    for i in range(len(unigram_counts)):
        for j in range(len(vocab)):
            unigram_counts[i][vocab[j]] = 0

    for i in range(len(documents_processed)):
        for w in documents_processed[i]:
            unigram_counts[i][w]+=1
    #finding portion of docs containing each word
    word_contained = []
    for i in range(len(vocab)):
        word_contained.append(0)
        for j in range(len(documents)):
            if unigram_counts[j][vocab[i]]>0:
                word_contained[i]+=1

    d = {}
    for j in range(len(vocab)):
        d[vocab[j]]=np.log10(len(documents)/(word_contained[j]+1))+1
        #(i,j) cell should have tf-idf score of ith doc and jth word in vocab

    #I was having a similar issue. Make sure youre using the formula from the wikipedia page for idf: idf = np.log10((N) / (n + 1)) + 1
    return d
    '''return 
    # This part is ungraded, however, to test your code, you'll need to implement this function
    # If you have implemented create_tfidf_matrix, this implementation should be straightforward
    documents_processed = []
    vocab_set = set()
    for i in range(len(documents)):
        documents_processed.append([])
        for w in documents[i]:
            wlow = w.lower()
            if not wlow in stopwords and w.isalnum():
                documents_processed[i].append(wlow)
                vocab_set.add(wlow)
    vocab = sorted(list(vocab_set))
    
    #counting words per doc
    unigram_counts = [{} for _ in range(len(documents))]
    for i in range(len(unigram_counts)):
        for j in range(len(vocab)):
            unigram_counts[i][vocab[j]] = 0

    for i in range(len(documents_processed)):
        for w in documents_processed[i]:
            unigram_counts[i][w]+=1
    #finding portion of docs containing each word
    word_contained = []
    for i in range(len(vocab)):
        word_contained.append(0)
        for j in range(len(documents)):
            if unigram_counts[j][vocab[i]]>0:
                word_contained[i]+=1

    d = {}
    for i in range(len(vocab)):
        d[vocab[i]] = (np.log10(len(documents)/(word_contained[i]+1))+1)
    
    return d'''

def calculate_sparsity(tfidf_matrix):
    total = 0
    zerocount = 0
    for v in np.nditer(tfidf_matrix):
        total+=1
        if v == 0:
            zerocount+=1
    return (zerocount/total)

def extract_salient_words(VT, vocabulary, k):
    ret = {}
    #iterate through 10 latent dimensions in VT
    for i in range(10):
        latent_dim = VT[i,:]
        row_vals = {}
        for j in range(len(vocabulary)):
            row_vals[vocabulary[j]]=latent_dim[j]
            top_k = sorted(row_vals, key=row_vals.get, reverse=True)[:k]
            ret[i] = top_k
    return ret

def get_similar_documents(U, Sigma, VT, doc_index, k):
    Sigma = np.diag(Sigma)
    ddoc = U[doc_index, :]
    similarities = {}
    for i in range(U.shape[0]):
        if i == doc_index:
            continue
        row = U[i, :]
        numerator = 0
        denom1 = 0
        denom2 = 0
        v1 = Sigma @ ddoc
        v2 = Sigma @ row
        for j in range(len(v1)):
            numerator += v1[j]*v2[j]
            denom1 += pow(v1[j],2)
            denom2 += pow(v2[j], 2)

        cos_sim = numerator/(sqrt(denom1)*sqrt(denom2))
        similarities[i] = cos_sim
    top_k = sorted(similarities, key=similarities.get, reverse=True)[:k]
    return top_k
    #REIMPLIMENT COS SIM MYSELF AND CHECK
    #IF NOT WORKING AT ALL, TRY PREV FUNCTION CODE EXCEPT FIX OUTPUT, SHOULD BE LIST NOT DICT

def document_retrieval(vocabulary, idf_values, U, Sigma, VT, query, k):
    q2 = []
    Sigma = np.diag(Sigma)
    #ignore query words not in vocab
    for w in query:
        wlow = w.lower()
        if wlow in vocabulary:
            q2.append(wlow)
    query_tfidf = np.zeros(shape=(len(vocabulary),))
    q2_word_counts = {}
    for w in vocabulary:
        q2_word_counts[w] = 0
    for w in q2:
        q2_word_counts[w] += 1
    for i in range(len(vocabulary)):
        query_tfidf[i] = q2_word_counts[vocabulary[i]]*idf_values[vocabulary[i]]
    
    #translate into low-dimensional space:
    qhat = np.linalg.inv(Sigma) @ VT @ query_tfidf

    similarities = {}
    for i in range(U.shape[0]):
        row = U[i, :]
        numerator = 0
        denom1 = 0
        denom2 = 0
        v1 = Sigma @ qhat
        v2 = Sigma @ row
        for j in range(len(v1)):
            numerator += v1[j]*v2[j]
            denom1 += pow(v1[j],2)
            denom2 += pow(v2[j], 2)

        cos_sim = numerator/(sqrt(denom1)*sqrt(denom2))
        similarities[i] = cos_sim
    top_k = sorted(similarities, key=similarities.get, reverse=True)[:k]
    return top_k

if __name__ == '__main__':
    
    tweets = []
    with open('data/covid-tweets-2020-08-10-2020-08-21.tokenized.txt', encoding="utf8") as f:
        tweets = [line.strip() for line in f.readlines()]

    stop_words = []
    with open('data/stop_words.txt') as f:
        stop_words = [line.strip() for line in f.readlines()]


    """Building Vector Space model using PMI"""
    '''
    frequent_unigrams = top_k_unigrams(tweets, stop_words, 1000)
    unigram_counter = top_k_unigrams(tweets, stop_words, -1)
    context_counter = context_word_frequencies(tweets, stop_words, 3, frequent_unigrams)

    word_vector = build_word_vector('ventilator', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'put': 6.301874856316369, 'patient': 6.222687002250096, 'tried': 6.158108051673095, 'wearing': 5.2564459708663875, 'needed': 5.247669358807432, 'spent': 5.230966480014661, 'enjoy': 5.177980198384708, 'weeks': 5.124941187737894, 'avoid': 5.107686157639801, 'governors': 5.103879572210065}

    word_vector = build_word_vector('mask', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'wear': 7.278203356425305, 'wearing': 6.760722107602916, 'mandate': 6.505074539073231, 'wash': 5.620700962265705, 'n95': 5.600353617179614, 'distance': 5.599542578641884, 'face': 5.335677912801717, 'anti': 4.9734651502193366, 'damn': 4.970725788331299, 'outside': 4.4802694058646}

    word_vector = build_word_vector('distancing', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'social': 8.637723567642842, 'guidelines': 6.244375965192868, 'masks': 6.055876420939214, 'rules': 5.786665161219354, 'measures': 5.528168931193456, 'wearing': 5.347796214635814, 'required': 4.896659865603407, 'hand': 4.813598338358183, 'following': 4.633301876715461, 'lack': 4.531964710683777}

    word_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'donald': 7.363071158640809, 'administration': 6.160023745590209, 'president': 5.353905139926054, 'blame': 4.838868198365827, 'fault': 4.833928177006809, 'calls': 4.685281547339574, 'gop': 4.603457978983295, 'failed': 4.532989597142956, 'orders': 4.464073158650432, 'campaign': 4.3804665561680824}

    word_vector = build_word_vector('pandemic', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'global': 5.601489175269805, 'middle': 5.565259949326977, 'amid': 5.241312533124981, 'handling': 4.609483077248557, 'ended': 4.58867551721951, 'deadly': 4.371399989758025, 'response': 4.138827482426898, 'beginning': 4.116495953781218, 'pre': 4.043655804452211, 'survive': 3.8777495603541254}

    word1_vector = build_word_vector('ventilator', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('covid-19', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.2341567704935342

    word2_vector = build_word_vector('mask', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.05127326904936171

    word1_vector = build_word_vector('president', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.7052644362543867

    word2_vector = build_word_vector('biden', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.6144272810573133

    word1_vector = build_word_vector('trudeau', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.37083874436657593

    word2_vector = build_word_vector('biden', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.34568665086152817
    '''
    """Exploring Word2Vec"""
    '''
    EMBEDDING_FILE = 'data/GoogleNews-vectors-negative300.bin.gz'
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

    similar_words =  get_most_similar(word2vec, 'ventilator', 3)
    print(similar_words)
    # [('respirator', 0.7864563465118408), ('mechanical_ventilator', 0.7063839435577393), ('intensive_care', 0.6809945702552795)]

    # Word analogy - Tokyo is to Japan as Paris is to what?
    print(word_analogy(word2vec, 'Tokyo', 'Japan', 'Paris'))
    # ('France', 0.7889978885650635)
    '''
    """Latent Semantic Analysis"""

    import nltk
    nltk.download('brown')
    from nltk.corpus import brown
    documents = [brown.words(fileid) for fileid in brown.fileids()]

    # Exploring the corpus
    print("The news section of the Brown corpus contains {} documents.".format(len(documents)))
    for i in range(3):
        document = documents[i]
        print("Document {} has {} words: {}".format(i, len(document), document))
    # The news section of the Brown corpus contains 500 documents.
    # Document 0 has 2242 words: ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]
    # Document 1 has 2277 words: ['Austin', ',', 'Texas', '--', 'Committee', 'approval', ...]
    # Document 2 has 2275 words: ['Several', 'defendants', 'in', 'the', 'Summerdale', ...]

    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stopwords_list = stopwords.words('english')

    # This will take a few minutes to run
    tfidf_matrix, vocabulary = create_tfidf_matrix(documents, stopwords_list)
    idf_values = get_idf_values(documents, stopwords_list)

    '''print(tfidf_matrix.shape)
    # (500, 40881)

    print(tfidf_matrix[np.nonzero(tfidf_matrix)][:5])
    # [5.96857651 2.1079054  3.         2.07572071 2.69897   ]

    print(vocabulary[2000:2010])
    # ['amoral', 'amorality', 'amorist', 'amorous', 'amorphous', 'amorphously', 'amortization', 'amortize', 'amory', 'amos']

    print(calculate_sparsity(tfidf_matrix))
    # 0.9845266994447298'''

    """SVD"""
    print('here1')
    U, Sigma, VT = randomized_svd(tfidf_matrix, n_components=10, n_iter=100, random_state=42)
    print('here2')

    '''salient_words = extract_salient_words(VT, vocabulary, 10)
    print(salient_words[1])
    # ['anode', 'space', 'theorem', 'v', 'q', 'c', 'p', 'operator', 'polynomial', 'af']'''

    print("We will fetch documents similar to document {} - {}...".format(3, ' '.join(documents[3][:50])))
    # We will fetch documents similar to document 3 - 
    # Oslo The most positive element to emerge from the Oslo meeting of North Atlantic Treaty Organization Foreign Ministers has been the freer , 
    # franker , and wider discussions , animated by much better mutual understanding than in past meetings . This has been a working session of an organization that...

    similar_doc_indices = get_similar_documents(U, Sigma, VT, 3, 5)
    for i in range(2):
        print("Document {} is similar to document 3 - {}...".format(similar_doc_indices[i], ' '.join(documents[similar_doc_indices[i]][:50])))
    # Document 61 is similar to document 3 - 
    # For a neutral Germany Soviets said to fear resurgence of German militarism to the editor of the New York Times : 
    # For the first time in history the entire world is dominated by two large , powerful nations armed with murderous nuclear weapons that make conventional warfare of the past...
    # Document 6 is similar to document 3 - 
    # Resentment welled up yesterday among Democratic district leaders and some county leaders at reports that Mayor Wagner had decided to seek a third term with Paul R. Screvane and Abraham D. Beame as running mates . 
    # At the same time reaction among anti-organization Democratic leaders and in the Liberal party... 
    
    query = ['Krim', 'attended', 'the', 'University', 'of', 'North', 'Carolina', 'to', 'follow', 'Thomas', 'Wolfe']
    print("We will fetch documents relevant to query - {}".format(' '.join(query)))
    relevant_doc_indices = document_retrieval(vocabulary, idf_values, U, Sigma, VT, query, 5)
    for i in range(2):
        print("Document {} is relevant to query - {}...".format(relevant_doc_indices[i], ' '.join(documents[relevant_doc_indices[i]][:50])))
    # Document 90 is relevant to query - 
    # One hundred years ago there existed in England the Association for the Promotion of the Unity of Christendom . 
    # Representing as it did the efforts of only unauthorized individuals of the Roman and Anglican Churches , and urging a communion of prayer unacceptable to Rome , this association produced little...
    # Document 101 is relevant to query - To what extent and in what ways did Christianity affect the United States of America in the nineteenth century ? ? 
    # How far and in what fashion did it modify the new nation which was emerging in the midst of the forces shaping the revolutionary age ? ? To what...
