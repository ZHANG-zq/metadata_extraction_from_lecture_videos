import pysrt
import stemmer
import numpy as np
import re
from pycorenlp import StanfordCoreNLP
import math
import json
import pickle
import copy
from mdlp import *
from nltk.stem import porter
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

punctuation = ['.']
pos_remain = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS']
# pos_remain = ['FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS']
# stopwords = ["a", "an", "and", "are", "as", "at", "be", "but", "by","for", "if", "in", "into", "is", "it","no", "not", "of", "on", "or", "such","that", "the", "their", "then", "there", "these","they", "this", "to", "was", "will", "with"]
stopwords = ["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"]
Mentioning = ['PRP', 'PRP$', 'WP', 'WP$']

NB_CLASSIFIER_PATH = "NB_Classifier.json"
NUM_TRAINING_DATA = 3
NLP_SERVER = None
eps = 1e-7
#Nounce = 0
DEFAULT_WINDOW_LEN = 50
stem_map = {}

def save_NB_Classifier(NB_Classifier):
    text = json.dumps(NB_Classifier)
    f = open(NB_CLASSIFIER_PATH, 'w')
    f.write(text)
    f.close()


def load_NB_Classifier():
    text = []
    for line in open(NB_CLASSIFIER_PATH):
        text.append(line)
    text = " ".join(text)
    NB_Classifier = json.loads(text)
    return NB_Classifier


def Classify(features, NB_Classifier):
    keyphrases = []
    num_phrases = 0
    log = {}
    f = features[features.keys()[0]]
    for phrase in f:
        num_phrases += 1
        log[phrase] = {}

    for i in xrange(num_phrases):
        key_prob = 0
        notkey_prob = 0
        for f_name, f in features.iteritems():
            phrase, v = f.items()[i]
            t = "%s_%d" % (f_name, v)
            # key_prob *= NB_Classifier['p_feature']['key'].get(t, eps / (NB_Classifier['num_keyphrases'] + eps*NB_Classifier['num_features']))
            # notkey_prob *= NB_Classifier['p_feature']['notkey'].get(t, eps / ((NB_Classifier['num_phrases']-NB_Classifier['num_keyphrases']) + eps*NB_Classifier['num_features']))
            key_prob += math.log(NB_Classifier['p_feature']['key'].get(t, eps / (
            NB_Classifier['num_keyphrases'] + eps * NB_Classifier['num_features'])))
            notkey_prob += math.log(NB_Classifier['p_feature']['notkey'].get(t, eps / (
            (NB_Classifier['num_phrases'] - NB_Classifier['num_keyphrases']) + eps * NB_Classifier['num_features'])))
        # key_prob *= NB_Classifier['p_class']['key']
        # notkey_prob *= NB_Classifier['p_class']['notkey']
        key_prob += math.log(NB_Classifier['p_class']['key'])
        notkey_prob += math.log(NB_Classifier['p_class']['notkey'])
        if key_prob > notkey_prob:
            keyphrases.append(phrase)
    return list(set(keyphrases))


def train_NB_Classifier(features, keyphrases):
    NB_Classifier = {'p_class': {'key': 0, 'notkey': 0}, 'p_feature': {'key': {}, 'notkey': {}}}
    num_keyphrases = 0.0
    num_phrases = 0.0
    num_features = len(features)

    f = features[features.keys()[0]]
    for i in xrange(len(f)):
        for phrase in f[i]:
            num_phrases += 1
            if phrase in keyphrases[i]:
                num_keyphrases += 1

    for f_name, f in features.iteritems():
        for i in xrange(len(f)):
            for phrase, v in f[i].iteritems():
                t = "%s_%d" % (f_name, v)
                if phrase in keyphrases[i]:
                    NB_Classifier['p_feature']['key'][t] = NB_Classifier['p_feature']['key'].get(t, 0.0) + 1
                    NB_Classifier['p_feature']['notkey'][t] = NB_Classifier['p_feature']['notkey'].get(t, 0.0)
                else:
                    NB_Classifier['p_feature']['notkey'][t] = NB_Classifier['p_feature']['notkey'].get(t, 0.0) + 1
                    NB_Classifier['p_feature']['key'][t] = NB_Classifier['p_feature']['key'].get(t, 0.0)

    # NB_Classifier['p_class']['key'] = ( num_keyphrases+eps ) / (num_phrases + eps*2)
    # NB_Classifier['p_class']['notkey'] = ( (num_phrases-num_keyphrases)+eps ) / (num_phrases + eps*2)
    NB_Classifier['p_class']['key'] = num_keyphrases / num_phrases
    NB_Classifier['p_class']['notkey'] = (num_phrases - num_keyphrases) / num_phrases

    for k, v in NB_Classifier['p_feature']['key'].iteritems():
        NB_Classifier['p_feature']['key'][k] = (v + eps) / (num_keyphrases + eps * num_features)
    for k, v in NB_Classifier['p_feature']['notkey'].iteritems():
        NB_Classifier['p_feature']['notkey'][k] = (v + eps) / ((num_phrases - num_keyphrases) + eps * num_features)

    NB_Classifier['num_features'] = num_features
    NB_Classifier['num_keyphrases'] = num_keyphrases
    NB_Classifier['num_phrases'] = num_phrases

    return NB_Classifier



def read_keyphrases(path):
    keyphrases = []
    stemmer = porter.PorterStemmer()
    for phrase in open(path):
        stem_phrase = []
        words = phrase.replace('\n', '').lower().split(" ")
        for w in words:
            stem_w = stemmer.stem_word(w)
            stem_phrase.append(stem_w)
        keyphrases.append(" ".join(stem_phrase))
    return keyphrases


def words_filter(text,  words, ori_words):
    count = 0
    reg = re.compile(r'([!"#%&()*+,-./:;<=>?@\[\\\]^_`{|}~])', re.IGNORECASE)
    text = re.sub(reg, '#', text)

    stemmer = porter.PorterStemmer()
    for word in text.split(" "):
        count += 1
        if word == '#' :
            ori_words.append("#")
        else:
            if word != " " and word != "":
                if word not in stem_map:
                    stem_map[word] = stemmer.stem_word(word)
                words.append(stem_map[word])
                ori_words.append(word)
    return count

def coref_pos_filter(text, words, ori_words):
    output = NLP_SERVER.annotate(text, properties={
        'annotators': 'coref',
        'outputFormat': 'json',
    })
    # The standard Lucene stopword
    stemmer = porter.PorterStemmer()
    count  = 0
    for sentence in output['sentences']:
        #print sentence['index']
        for word in sentence['tokens']:
            count += 1
            if word['pos'] in ['SYM']:
                ori_words.append('#')
            else:
                if word['word'] not in stem_map:
                    stem_map[word['word']] = stemmer.stem_word(word['word'])
                words.append(stem_map[word['word']])
                ori_words.append(word['word'])
    return count

def candidate_words(training_data_path):
    sentences_time = []
    chunk = [[]]
    words = []
    ori_words = []
    num_words_1min = 0
    pos = 0
    num_word = []
    subs = pysrt.open(training_data_path)
    for sub in subs:
        text = []
        for w in sub.text.split(' '):
            try:
                text.append(w.encode('ascii', 'ignore').lower())
            except UnicodeEncodeError, e:
                print e
        text = ' '.join(text)
        count = coref_pos_filter(text.replace('\n', ' ').lower(), words,ori_words)
        #count = words_filter(sub.text.replace('\n', ' ').decode('utf-8').encode('ascii', 'ignore').lower(), words, ori_words, stem_map)

        starttime = sub.start.milliseconds + sub.start.seconds * 1000 + sub.start.minutes * 60000 + sub.start.hours * 60 * 60000
        endtime = sub.end.milliseconds + sub.end.seconds * 1000 + sub.end.minutes * 60000 + sub.end.hours * 60 * 60000
        sentences_time.append([(pos,pos+count),(starttime, endtime)])

        num_words_1min += count
        pos += count
        # sentences_time.append([sub.start,sub.end])
        if sub.start.minutes != sub.end.minutes or sub.end.seconds == 0:
            num_word.append(num_words_1min)
            num_words_1min = 0
    if num_words_1min != 0:
        num_word.append(num_words_1min)

    # For Localspan feature : get the average number of words spoken by the lecturer for a time span of five minutes
    num_span = len(num_word) / 5
    num_word_5min = []
    for t in range(num_span):
        i, j = t * 5, (t + 1) * 5
        num_word_5min.append(sum(num_word[i:j]))
    num_word_5min = sum(num_word_5min) / len(num_word_5min)

    return words, ori_words, stem_map, num_word_5min, sentences_time


def audio_candidate_phrase(training_data_path):
    words, ori_words, stem_map, num_word_5min, sentences_time = candidate_words(training_data_path)
    windowLength = 4
    phrases = {}

    for i in range(len(ori_words)):
        for j in range(windowLength):
            if i + j + 1 > len(ori_words):
                continue

            b_skip = False
            # filter out undesire n-gram
            if ori_words[i] == "#":
                break
            for w in ori_words[i:i + j + 1]:
                if re.search(r'[^0-9a-zA-Z]',w):  # Only non-alphanumeric characters that were not present in any keyword in the training set were removed (keeping e.g., C++).
                    b_skip = True
                    break
                elif not re.search(r'[a-zA-Z]',w):  # Numbers were removed only if they stood separately (keeping e.g., 4YourSoul.com).
                    b_skip = True
                    break
            if b_skip:
                break
            if j == 0 and ori_words[i] in stopwords:
                break
            stem_p = []
            for w in ori_words[i:i + j + 1]:
                stem_p.append(stem_map[w])
            stem_p = ' '.join(stem_p)
            if stem_p not in phrases:
                phrases[stem_p] = ori_words[i:i + j + 1]

    return words, phrases, ori_words, num_word_5min, sentences_time


def print_phrase(phrases):
    for stem_p, p in phrases.iteritems():
        print stem_p


def Get_ATD(words, phrases, k):
    windowLength = int(math.ceil(len(words) / float(k)))
    phrase_count = {}
    frequency = {}
    mean = {}
    variance = {}
    atd = {}
    for stem_p in phrases:
        frequency[stem_p] = np.zeros([1, k], float)[0]

    for w in range(k):
        i = w * windowLength
        j = (w + 1) * windowLength
        for stem_p in phrases.iterkeys():
            frequency[stem_p][w] = get_frequency(words[i:j + len(stem_p) - 1], stem_p.split(' '))

    for k, v in frequency.iteritems():
        phrase_count[k] = np.sum(v)
        mean[k] = np.sum(v) / float(windowLength)
        assert (mean[k] != 0)

    for k, v in frequency.iteritems():
        variance[k] = np.sum(np.power((v - mean[k]), 2)) / float(windowLength)

    for k in frequency.iterkeys():
        atd[k] = mean[k] / (mean[k] + variance[k])

    return atd, phrase_count


def print_feature(feature):
    for phrase in feature:
        if feature[phrase] == 0:
            continue
        print phrase, feature[phrase]


def get_frequency(words, phrase):
    beg = 0
    count = 0.0
    words_len = len(words)
    phrase_len = len(phrase)
    while beg <= words_len - phrase_len:
        b_match = True
        for i in range(phrase_len):
            if words[beg + i] != phrase[i]:
                b_match = False
                break
        if b_match:
            count += 1
            beg += phrase_len
        else:
            beg += 1
    return count


def Get_Localspan(words, phrases, num_word_5min):
    k = int(math.ceil(len(words) / float(num_word_5min)))
    frequency = {}
    for stem_p in phrases:
        frequency[stem_p] = np.zeros([1, k], float)[0]

    for w in range(k):
        i = w * num_word_5min
        j = (w + 1) * num_word_5min
        if j > len(words):
            # i = len(words) - num_word_5min - len(stem_p) + 1
            i = len(words) - num_word_5min + 1
            j = len(words)
        print i, j, j - i
        for stem_p in phrases.iterkeys():
            frequency[stem_p][w] = get_frequency(words[i:j + len(stem_p) - 1], stem_p.split(' '))

    Localspan = {}
    for stem_p in phrases.iterkeys():
        Localspan[stem_p] = np.max(frequency[stem_p] - np.sum(frequency[stem_p]) / k)

    return Localspan


def Get_C_Value(words, phrases, phrase_count):
    C_Value = {}
    print len(phrases)
    for phrase in phrases.iterkeys():
        corpus = [phrase]
        p = phrase.split(' ')
        if len(p) != 3:
            for bigger_phrase in phrases.iterkeys():
                b_p = bigger_phrase.split(' ')
                if len(b_p) > len(p):
                    if contain(b_p, p):
                        corpus.append(bigger_phrase)
                        # print phrase + '  <contained by>  '+ bigger_phrase

        C_Value[phrase] = math.log(len(p), 2)
        if len(corpus) == 1:
            C_Value[phrase] *= phrase_count[phrase]
            # print phrase, C_Value[phrase], math.log( len(p), 2), phrase_count[phrase]
        else:
            C_Value[phrase] *= (phrase_count[phrase] - sum([phrase_count[t] for t in corpus]) / float(len(corpus)))
            # print phrase, C_Value[phrase], math.log(len(p), 2), phrase_count[phrase], sum([phrase_count[t] for t in corpus]), float(len(corpus))

    return C_Value


def contain(bigger_phrase, phrase):
    i = 0
    while i + len(phrase) <= len(bigger_phrase):
        b_match = True
        for j in range(len(phrase)):
            if bigger_phrase[j + i] != phrase[j]:
                b_match = False
                break
        if b_match:
            return True
        i += 1
    return False


def Get_IDF(docs):
    IDF = {'GD': float(len(docs)), 'NOD': {}}
    for i in range(len(docs)):
        visited = []
        for phrase in docs[i]:
            if phrase not in visited:
                IDF['NOD'][phrase] = IDF['NOD'].get(phrase, 0.0) + 1
                visited.append(phrase)
    return IDF


def Get_TF_IDF(doc, IDF):
    TF = {}
    for phrase in doc:
        TF[phrase] = TF.get(phrase, 0.0) + 1
    TF_IDF = {}
    for k, v in TF.iteritems():
        TF_IDF[k] = v / len(doc) * math.log((IDF['GD'] + 1) / (IDF['NOD'].get(k, 0.0) + 1), 2)
    return TF_IDF


def Get_Cuewords(ori_words, stem_map, phrases):
    Cuewords = {}
    for stem_p in phrases:
        Cuewords[stem_p] = 0
    for i in range(len(ori_words) - 2):
        if ori_words[i] != 'called' and ori_words[i] != 'defined':
            continue
        if ori_words[i + 1] != 'as':
            continue
        for stem_p in phrases.iterkeys():
            p = stem_p.split(' ')
            if len(ori_words) - 2 - i < len(p):
                continue
            b_match = True
            for j in range(len(p)):
                if stem_map[ori_words[i + 2 + j]] != p[j]:
                    b_match = False
                    break
            if b_match:
                Cuewords[stem_p] += 1
    return Cuewords


def check_features(features, phrases):
    # test1
    for i in xrange(1, len(features)):
        assert (len(features[i]) == len(features[i - 1]))



def discretize(feature, keyphrases, feature_name, mode):
    if mode == 'train':
        feature_discretized = []
        length = sum([len(feature[i]) for i in xrange(len(feature))])

        X = np.zeros(length)
        y = np.zeros(length)
        idx = 0
        for i in xrange(len(feature)):
            feature_discretized.append({})
            for j in xrange(len(feature[i])):
                X[idx] = feature[i][feature[i].keys()[j]]
                if feature[i].keys()[j] in keyphrases[i]:
                    y[idx] = 1.0
                    print "find keyphrase"
                idx += 1
        mdlp = MDLP()

        X_cutpoint = mdlp.cut_points(X, y)
        X_discrete = mdlp.discretize_feature(X, X_cutpoint)

        idx = 0
        for i in xrange(len(feature)):
            for j in xrange(len(feature[i])):
                feature_discretized[i][feature[i].keys()[j]] = X_discrete[idx]
                idx += 1

        pickle.dump(X_cutpoint, open('data/%s_cutpoint' % feature_name, 'w'))
        print X_cutpoint
        print '__________________________________________________________________________________________________'

    if mode == 'test':
        length = len(feature)
        X = np.zeros(length)
        for i in xrange(length):
            X[i] = feature[feature.keys()[i]]

        mdlp = MDLP()
        X_cutpoint = pickle.load(open('data/%s_cutpoint' % feature_name, 'r'))
        X_discrete = mdlp.discretize_feature(X, X_cutpoint)

        feature_discretized = {}
        for i in xrange(length):
            feature_discretized[feature.keys()[i]] = X_discrete[i]
    return feature_discretized


def plot_fig(x, score, heading, fig_no):
    fig = plt.figure(fig_no, figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, score, label=heading)
    ax.legend()
    plt.show()


def Count_phrase_in_sentence(phrase, sentence):
    len_phrase = len(phrase)
    len_sen = len(sentence)
    count = 0
    for i in xrange(0, len_sen - len_phrase + 1):
        b_found = True
        for j in xrange(len_phrase):
            if phrase[j] != sentence[i + j]:
                b_found = False
        if b_found == True:
            count += 1

    return count

def text_tiling_SegmentInSentence(words, keyphrases, sentences_time, sen_len, fig_no):
    sentences_EndIdx = []
    for st in sentences_time:
        sentences_EndIdx.append(st[0][1])
    # Porter Stemming and removing stop words
    sentences = []
    s = []
    i = 0
    j = 0
    for word in words:
        i += 1
        j += 1
        s.append(word)
        if i in sentences_EndIdx and j >= sen_len:
            j = 0
            # -1 is to prevent the whitespace that is appended at the end to be included in the sentence
            sentences.append(s)
            s = []
    # If the last sentence is of length less than sen_length
    if s != []:
        sentences.append(s)

    # Vectorizing Sentences using Sklearn to determine Cosine Similarity between adjacent sentences
    # tfidf_vectorizer = TfidfVectorizer()
    # tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

    matrix = np.zeros([len(sentences), len(keyphrases)])
    frequency = np.zeros([len(sentences), len(keyphrases)])
    NOS = np.zeros(len(keyphrases))
    for i in xrange(len(keyphrases)):
        for j in xrange(len(sentences)):
            count = Count_phrase_in_sentence(keyphrases[i].split(' '), sentences[j])
            frequency[j, i] = count
            if count > 0:
                NOS[i] += 1
                # assert(NOS[i] != 0)
    matrix = frequency / np.log((len(sentences) + eps) / (NOS + eps))

    score = []
    for i in range(0, matrix.shape[0] - 2):
        score.append(cosine_similarity(matrix[i:i + 1], matrix[i + 1:i + 2])[0][0])

    # Plotting Cosine Similarity
    plot_fig(range(1, matrix.shape[0] - 1), score, 'Lexical Similarity with Sentence Length ' + str(sen_len),
             fig_no)

    # Implementing Windowdiff measure
    mean_score = np.mean(score)
    std_score = np.std(score)
    # Threshold is defined as Mean Score - Standard Deviation
    threshold = mean_score - std_score
    boundary = []
    for i in range(0, len(score) - 2):
        # score[0] represents the cosine similarity between sentence 1 and sentence 2, score[1] between 2 and 3 and score[2] between 3 and 4
        # If depth is greater than threshold, then there will be a dissimilariy between sentence 2 and sentence 3, so we are marking sentence 2 as the boundary
        depth = score[i] - score[i + 1] + score[i + 2] - score[i + 1]
        if depth >= threshold:
            boundary.append(i + 1)  # Storing positions of Sentences that represent a boundary
    # Replacing boundaries with 1 and words with 0
    segmented_words = []
    one_segment_words = []
    for i in range(0, len(sentences)):
        one_segment_words += sentences[i]
        if i in boundary:
            segmented_words.append(one_segment_words)
            one_segment_words = []

    return segmented_words

def text_tiling(words, keyphrases, sen_len, fig_no):
    # Porter Stemming and removing stop words
    sentences = []
    s = []
    j = 0
    for word in words:
        j = j + 1
        s.append(word)
        if j == sen_len:
            # -1 is to prevent the whitespace that is appended at the end to be included in the sentence
            sentences.append(s)
            s = []
            j = 0
    # If the last sentence is of length less than sen_length
    if s != []:
        sentences.append(s)

    # Vectorizing Sentences using Sklearn to determine Cosine Similarity between adjacent sentences
    # tfidf_vectorizer = TfidfVectorizer()
    # tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

    matrix = np.zeros([len(sentences), len(keyphrases)])
    frequency = np.zeros([len(sentences), len(keyphrases)])
    NOS = np.zeros(len(keyphrases))
    for i in xrange(len(keyphrases)):
        for j in xrange(len(sentences)):
            count = Count_phrase_in_sentence(keyphrases[i].split(' '), sentences[j])
            frequency[j, i] = count
            if count > 0:
                NOS[i] += 1
                # assert(NOS[i] != 0)
    matrix = frequency / np.log((len(sentences) + eps) / (NOS + eps))

    score = []
    for i in range(0, matrix.shape[0] - 2):
        score.append(cosine_similarity(matrix[i:i + 1], matrix[i + 1:i + 2])[0][0])

    # Plotting Cosine Similarity
    plot_fig(range(1, matrix.shape[0] - 1), score, 'Lexical Similarity with Sentence Length ' + str(sen_len),
             fig_no)

    # Implementing Windowdiff measure
    mean_score = np.mean(score)
    std_score = np.std(score)
    # Threshold is defined as Mean Score - Standard Deviation
    threshold = mean_score - std_score
    boundary = []
    for i in range(0, len(score) - 2):
        # score[0] represents the cosine similarity between sentence 1 and sentence 2, score[1] between 2 and 3 and score[2] between 3 and 4
        # If depth is greater than threshold, then there will be a dissimilariy between sentence 2 and sentence 3, so we are marking sentence 2 as the boundary
        depth = score[i] - score[i + 1] + score[i + 2] - score[i + 1]
        if depth >= threshold:
            boundary.append(i + 1)  # Storing positions of Sentences that represent a boundary
    # Replacing boundaries with 1 and words with 0
    segmented_words = []
    one_segment_words = []
    for i in range(0, len(sentences)):
        one_segment_words += sentences[i]
        if i in boundary:
            segmented_words.append(one_segment_words)
            one_segment_words = []

    return segmented_words


#  video part

def candidate_video_words(training_data_path):
    ppt_time = []
    words = []
    ori_words = []
    words_height = []
    f_obj = open(training_data_path,'r')
    lines = f_obj.readlines()
    num_slide = int(lines[0])
    idx = 1
    stemmer = porter.PorterStemmer()
    for i in xrange(num_slide):
        slide_name = lines[idx]
        for item in lines[idx + 1].split(','):
            try:
                item = item.encode('ascii', 'ignore').lower()
            except UnicodeDecodeError, e:
                continue
            item = item.split('&')
            if item[0] == '#':
                ori_words.append(item[0])
                words_height.append(1e8)
            else:
                if item[0] not in stem_map:
                    stem_map[item[0]] = stemmer.stem_word(item[0])
                words.append(stem_map[item[0]])
                ori_words.append(item[0])
                words_height.append(float(item[1]))
        idx += 2
    return words, ori_words, words_height, ppt_time


def video_candidate_phrase(training_data_path):
    words, ori_words, words_height, ppt_time = candidate_video_words(training_data_path)
    windowLength = 4
    phrases = {}
    PH = {}
    min_PH = 1e8
    for i in range(len(ori_words)):
        if ori_words[i] in stopwords:
            continue
        for j in range(windowLength):
            if ori_words[i + j] in stopwords:
                continue

            if i + j + 1 > len(ori_words):
                break

            b_skip = False
            # filter out undesire n-gram
            if ori_words[i] == "#":
                break
            for w in ori_words[i:i + j + 1]:
                if re.search(r'[^0-9a-zA-Z]',w):  # Only non-alphanumeric characters that were not present in any keyword in the training set were removed (keeping e.g., C++).
                    b_skip = True
                    break
                elif not re.search(r'[a-zA-Z]',w):  # Numbers were removed only if they stood separately (keeping e.g., 4YourSoul.com).
                    b_skip = True
                    break
            if b_skip:
                break

            stem_p = []
            for w in ori_words[i:i + j + 1]:
                stem_p.append(stem_map[w])
            stem_p = ' '.join(stem_p)
            if stem_p not in phrases:
                phrases[stem_p] = ori_words[i:i + j + 1]
            PH[stem_p] = max(min(words_height[i:i + j + 1]), PH.get(stem_p,0))
            if min_PH > PH[stem_p]:
                min_PH = PH[stem_p]

    for k,v in PH.iteritems():
        PH[k] = v/min_PH
        #print '%s : %f' % (k,PH[k])

    print "\n\n\n__________________________________________________"
    testPH = copy.copy(PH)
    for i in xrange(20):
        t = np.array(testPH.values())
        idx = np.argmax(t)
        print testPH.keys()[idx],testPH[testPH.keys()[idx]]
        testPH[testPH.keys()[idx]] = 0
    print "__________________________________________________\n\n\n"
    return words, phrases, ori_words, PH, ppt_time



if __name__ == '__main__':
    NLP_SERVER = StanfordCoreNLP('http://localhost:9000')
    mode = 'train'
    if mode == 'train':
        words_all, phrases_all, ori_words_all, num_word_5min_all, keyphrases_all = [], [], [], [], []

        # for i in xrange(NUM_TRAINING_DATA):
        #     training_data_path = "data/train%d" % (i + 1)
        #     words_t, phrases_t, ori_words_t, num_word_5min_t, sentences_time_t = audio_candidate_phrase(training_data_path)
        #     keyphrases_t = read_keyphrases(training_data_path + "_keyphrases")
        #     words_all.append(words_t)
        #     phrases_all.append(phrases_t)
        #     ori_words_all.append(ori_words_t)
        #     num_word_5min_all.append(num_word_5min_t)
        #     keyphrases_all.append(keyphrases_t)
        # IDF = Get_IDF(phrases_all)
        # ATD_all, Localspan_all, C_Value_all, Cuewords_all, TF_IDF_all = [], [], [], [], []
        # for i in xrange(NUM_TRAINING_DATA):
        #     words, phrases, ori_words, num_word_5min, keyphrases = words_all[i], phrases_all[i], ori_words_all[i], num_word_5min_all[i], keyphrases_all[i]
        #     ATD, phrase_count = Get_ATD(words, phrases, DEFAULT_WINDOW_LEN)
        #     Localspan = Get_Localspan(words, phrases, num_word_5min)
        #     C_Value = Get_C_Value(words, phrases, phrase_count)
        #     Cuewords = Get_Cuewords(ori_words, stem_map, phrases)
        #     TF_IDF = Get_TF_IDF(phrases, IDF)
        #     check_features([ATD, Localspan, C_Value, Cuewords, TF_IDF], phrases)
        #     ATD_all.append(ATD)
        #     Localspan_all.append(Localspan)
        #     C_Value_all.append(C_Value)
        #     Cuewords_all.append(Cuewords)
        #     TF_IDF_all.append(TF_IDF)
        # ATD_all = discretize(ATD_all, keyphrases_all, 'ATD', 'train')
        # Localspan_all = discretize(Localspan_all, keyphrases_all, 'Localspan', 'train')
        # C_Value_all = discretize(C_Value_all, keyphrases_all, 'C_Value', 'train')
        # Cuewords_all = discretize(Cuewords_all, keyphrases_all, 'Cuewords', 'train')
        # TF_IDF_all = discretize(TF_IDF_all, keyphrases_all, 'TF_IDF', 'train')
        #
        # Audio_NB_Classifier = train_NB_Classifier({'ATD': ATD_all, 'Localspan': Localspan_all, 'C_Value': C_Value_all, 'Cuewords': Cuewords_all,'TF_IDF': TF_IDF_all}, keyphrases_all)
        # pickle.dump(Audio_NB_Classifier, open('data/OBJ_Audio_NB_Classifier', 'w'))
        # pickle.dump(IDF, open('data/OBJ_IDF', 'w'))
        # print Audio_NB_Classifier


        # video part
        words_all, phrases_all, ori_words_all, PH_all, keyphrases_all = [], [], [], [], []
        for i in xrange(NUM_TRAINING_DATA):
            training_data_path = "data/train%d_slide/slide" % (i + 1)
            words_t, phrases_t, ori_words_t, PH_t, ppt_time_t = video_candidate_phrase(training_data_path)
            keyphrases_t = read_keyphrases("data/train%d_keyphrases" % (i + 1))
            words_all.append(words_t)
            phrases_all.append(phrases_t)
            ori_words_all.append(ori_words_t)
            PH_all.append(PH_t)
            keyphrases_all.append(keyphrases_t)

        MOR_all, COR_all = [], []
        for i in xrange(NUM_TRAINING_DATA):
            words_t, phrases_t, ori_words_t, keyphrases_t = words_all[i], phrases_all[i], ori_words_all[i], keyphrases_all[i]
            # MOR = Get_MOR(words, phrases, phrase_count)
            # COR = Get_COR(ori_words, stem_map, phrases)
            # MOR_all.append(MOR)
            # COR_all.append(COR)
        PH_all = discretize(PH_all, keyphrases_all, 'PH', 'train')
        # MOR_all = discretize(Localspan_all, keyphrases_all, 'MOR', 'train')
        # COR_all = discretize(C_Value_all, keyphrases_all, 'COR', 'train')

        #Audio_NB_Classifier = train_NB_Classifier({'PH': PH_all, 'MOR': MOR_all, 'COR': COR_all}, keyphrases_all)
        video_NB_Classifier = train_NB_Classifier({'PH': PH_all}, keyphrases_all)
        pickle.dump(video_NB_Classifier, open('data/OBJ_video_NB_Classifier', 'w'))
        print video_NB_Classifier


    else:
        video_NB_Classifier = pickle.load(open('data/OBJ_video_NB_Classifier', 'r'))
        training_data_path = "data/test1"
        words, phrases, ori_words, PH, ppt_time = video_candidate_phrase( training_data_path + '_slide/slide' )

        PH = discretize(PH, None, 'PH', 'test')
        keyphrases = Classify(
            {'PH': PH}, video_NB_Classifier)
        print keyphrases
