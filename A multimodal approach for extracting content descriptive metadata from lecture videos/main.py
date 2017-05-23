import pysrt
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
# pos_remain = ['FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS']
# stopwords = ["a", "an", "and", "are", "as", "at", "be", "but", "by","for", "if", "in", "into", "is", "it","no", "not", "of", "on", "or", "such","that", "the", "their", "then", "there", "these","they", "this", "to", "was", "will", "with"]
# stopwords = ["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"]
stopwords = []
POS_TAG = {'NOUN':['NN', 'NNS', 'NNP', 'NNPS'],'ADJ':['JJ', 'JJR', 'JJS','VBD','VBG','VBN'],'ADV':['RB','RBR','RBS'],'IN' : ['IN'],'DT' : ['DT'],'NOT_NP' : ['VBZ', 'VB', 'VBP', 'VBZ','WP','WDT','WRB','WP$','PRP','PRP$','RP','TO','CC','CD','FW','UH','MD','LS','EX','SYM']}


NB_CLASSIFIER_PATH = "NB_Classifier.json"
NUM_TRAINING_DATA = 3
NLP_SERVER = None
eps = 1e-7
DEFAULT_WINDOW_LEN = 50
stem_map = {}
sentences_time = []
ppts_time = []
segments_time = []

def read_stopwords():
    for line in open('stoplist.txt'):
        if line.find('#') == -1:
            stopwords.append(line)

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
    probability = {}
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
            probability[phrase] = key_prob

    return list(set(keyphrases)), probability


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



def read_keyphrases(path,phrases):
    #exclude the keyphrases which do not appear in the phrases we ocr

    keyphrases = []
    stemmer = porter.PorterStemmer()
    for phrase in open(path):
        stem_phrase = []
        words = phrase.replace('\n', '').lower().split(" ")
        for w in words:
            stem_w = stemmer.stem(w)
            stem_phrase.append(stem_w)

        keyphrase = " ".join(stem_phrase)
        if keyphrase not in phrases:
            print "keyphrase : %s doesn't appear in phrases" % keyphrase
        else:
            keyphrases.append(keyphrase)

    return keyphrases



def coref_pos_filter(text, words_pos, ori_words):
    output = NLP_SERVER.annotate(text, properties={
        'annotators': 'coref',
        'outputFormat': 'json',
        'timeout': 300000,
    })
    # The standard Lucene stopword
    stemmer = porter.PorterStemmer()
    count  = 0
    for sentence in output['sentences']:
        #print sentence['index']
        for word in sentence['tokens']:
            #print word['word'],  word['pos'],
            count += 1
            if word['pos'] in ['SYM']:
                ori_words.append(u'.')
                words_pos.append(u'.')
                stem_map[u'.'] = u'.'
            else:
                if word['word'] not in stem_map:
                    stem_map[word['word']] = stemmer.stem(word['word'])
                words_pos.append(word['pos'])
                ori_words.append(word['word'])

    return count

def candidate_audio_words(training_data_path):

    chunk = [[]]
    words_pos = []
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
        count = coref_pos_filter(text.replace('\n', ' ').lower(), words_pos, ori_words)

        starttime = sub.start.milliseconds + sub.start.seconds * 1000 + sub.start.minutes * 60000 + sub.start.hours * 60 * 60000
        endtime = sub.end.milliseconds + sub.end.seconds * 1000 + sub.end.minutes * 60000 + sub.end.hours * 60 * 60000
        #sentences_time.append([(pos,pos+count),(starttime, endtime)])
        #print 'sentence : ',ori_words[pos:pos+count]
        sentences_time.append({'pos':pos, 'len': count, 'starttime':starttime, 'endtime':endtime})

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

    return words_pos, ori_words, num_word_5min



def print_phrase(phrases):
    for stem_p, p in phrases.iteritems():
        print stem_p


def Get_ATD(ori_words, phrases, k):
    windowLength = int(math.ceil(len(ori_words) / float(k)))
    frequency = {}
    mean = {}
    variance = {}
    atd = {}
    for stem_p in phrases:
        frequency[stem_p] = np.zeros([1, k], float)[0]

    for w in range(k):
        i = w * windowLength
        j = (w + 1) * windowLength
        for stem_p in phrases.keys():
            frequency[stem_p][w] = get_frequency(ori_words[i:j + len(stem_p) - 1], stem_p.split(' '))

    for k, v in frequency.iteritems():
        #phrase_count[k] = np.sum(v)    not accurate, because there is some window overlap
        mean[k] = np.sum(v) / float(windowLength)
        assert (mean[k] != 0)

    for k, v in frequency.iteritems():
        variance[k] = np.sum(np.power((v - mean[k]), 2)) / float(windowLength)

    for k in frequency.iterkeys():
        atd[k] = mean[k] / (mean[k] + variance[k])

    return atd


def print_feature(feature):
    for phrase in feature:
        if feature[phrase] == 0:
            continue
        print phrase, feature[phrase]


def get_frequency(ori_words, stem_phrase):
    beg = 0
    count = 0.0
    words_len = len(ori_words)
    phrase_len = len(stem_phrase)
    while beg <= words_len - phrase_len:
        b_match = True
        for i in range(phrase_len):
            if stem_map[ori_words[beg + i]] != stem_phrase[i]:
                b_match = False
                break
        if b_match:
            count += 1
            beg += phrase_len
        else:
            beg += 1
    return count


def Get_Localspan(ori_words, phrases, num_word_5min):
    k = int(math.ceil(len(ori_words) / float(num_word_5min)))
    frequency = {}
    for stem_p in phrases:
        frequency[stem_p] = np.zeros([1, k], float)[0]

    for w in range(k):
        i = w * num_word_5min
        j = (w + 1) * num_word_5min
        if j > len(ori_words):
            # i = len(words) - num_word_5min - len(stem_p) + 1
            i = len(ori_words) - num_word_5min + 1
            j = len(ori_words)
        print i, j, j - i
        for stem_p in phrases.iterkeys():
            frequency[stem_p][w] = get_frequency(ori_words[i:j + len(stem_p) - 1], stem_p.split(' '))

    Localspan = {}
    for stem_p in phrases.iterkeys():
        Localspan[stem_p] = np.max(frequency[stem_p] - np.sum(frequency[stem_p]) / k)

    return Localspan


def Get_C_Value(ori_words, phrases, phrase_count):
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


def Get_Cuewords(ori_words, phrases):
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


def Get_audio_words_time(word_idx):
    for st in sentences_time:
        if st['pos'] <= word_idx < (st['pos']+st['len']):
            print 'sentence time : ', st['starttime'], ' --> ', st['endtime']
            print st['pos'], word_idx, (st['pos']+st['len'])
            print 'segment time : ', st['starttime'] + (st['endtime']-st['starttime']) * ((word_idx-st['pos'])/float(st['len']))
            return st['starttime'] + (st['endtime']-st['starttime']) * ((word_idx-st['pos'])/float(st['len']))
    assert(False)

def text_tiling_slidingwindow(words, keyphrases, window_size, sliding_interval):
    windows_pos = []
    for i in xrange(0,len(words),sliding_interval):
        windows_pos.append( [ i, min( i+window_size, len(words) ) ] )
        print 'window : ', i, i + window_size
        if windows_pos[-1][1] == len(words):
            break


    matrix = np.zeros([len(windows_pos), len(keyphrases)])
    frequency = np.zeros([len(windows_pos), len(keyphrases)])
    NOS = np.zeros(len(keyphrases))
    for i in xrange(len(keyphrases)):
        for j in xrange(len(windows_pos)):
            count = Count_phrase_in_sentence(keyphrases[i].split(' '), words[windows_pos[j][0]:windows_pos[j][1]])
            frequency[j, i] = count
            if count > 0:
                NOS[i] += 1
                # assert(NOS[i] != 0)
    matrix = frequency / np.log((len(windows_pos) + eps) / (NOS + eps))

    score = []
    for i in range(0, matrix.shape[0] - 2):
        score.append(cosine_similarity(matrix[i:i + 1], matrix[i + 1:i + 2])[0][0])

    # Plotting Cosine Similarity
    plot_fig(range(1, matrix.shape[0] - 1), score, 'Lexical Similarity with Sentence Length ' + str(500),fig_no=1)

    # Implementing Windowdiff measure
    mean_score = np.mean(score)
    std_score = np.std(score)
    # Threshold is defined as Mean Score - Standard Deviation
    threshold = mean_score - 17*std_score
    boundary = []
    for i in range(0, len(score) - 2):
        depth = (score[i] - score[i + 1]) + (score[i + 2] - score[i + 1])
        if depth >= threshold:
            boundary.append(i + 1)  # Storing positions of Sentences that represent a boundary
    # Replacing boundaries with 1 and words with 0
    segmented_words = []
    boundary = [0] + boundary
    for i in xrange(1,len(boundary)):
        segments_time.append({})
        segments_time[-1]['pos'], segments_time[-1]['len'] = windows_pos[boundary[i-1]][0], windows_pos[boundary[i]][0]-windows_pos[boundary[i-1]][0]
        segmented_words.append(words[ segments_time[-1]['pos'] : segments_time[-1]['pos']+segments_time[-1]['len'] ])
        segments_time[-1]['starttime'] = Get_audio_words_time( segments_time[-1]['pos'] )
        segments_time[-1]['endtime'] = Get_audio_words_time( segments_time[-1]['pos']+segments_time[-1]['len'] )

    return segmented_words


def align_segments_time():
    aligned_segments_time = copy.copy(segments_time)
    closest_pt = (-1,1e10)
    threshold  = 60000 *2 # 60 seconds
    last_j = 0
    for i in xrange(len(segments_time)):
        for j in xrange(last_j, len(ppts_time)):
            if abs( ppts_time[j]['starttime'] - segments_time[i]['starttime'] ) < closest_pt[1]:
                closest_pt = (j, abs( ppts_time[j]['starttime'] - segments_time[i]['starttime'] ))
            else:
                if closest_pt[1] < threshold :
                    if i > 0:
                        aligned_segments_time[i-1]['endtime'] = ppts_time[closest_pt[0]]['starttime']
                    aligned_segments_time[i]['starttime'] = ppts_time[closest_pt[0]]['starttime']
                    last_j = j-1   # merge two segment that close to the same slide
                    # last_j = j # don't merge two segment that close to the same slide
                else:
                    last_j = j-1
                closest_pt = (-1, 1e10)
                break

    merged_aligned_segments_time = []
    b_merge = False
    for i in xrange(len(aligned_segments_time)):
        if aligned_segments_time[i]['starttime']== aligned_segments_time[i]['endtime']:
            assert( i != len(aligned_segments_time)-1 )
            if b_merge:
                merged_aligned_segments_time[-1]['len'] += aligned_segments_time[i]['len']
            else:
                merged_aligned_segments_time.append(copy.copy(aligned_segments_time[i]))
            b_merge = True
        else:
            if b_merge:
                merged_aligned_segments_time[-1]['len'] += aligned_segments_time[i]['len']
                merged_aligned_segments_time[-1]['endtime'] = aligned_segments_time[i]['endtime']
            else:
                merged_aligned_segments_time.append(copy.copy(aligned_segments_time[i]))
            b_merge = False

    return merged_aligned_segments_time


# def text_tiling(words, keyphrases, sen_len, fig_no):
#     # Porter Stemming and removing stop words
#     sentences = []
#     s = []
#     j = 0
#     for word in words:
#         j = j + 1
#         s.append(word)
#         if j == sen_len:
#             # -1 is to prevent the whitespace that is appended at the end to be included in the sentence
#             sentences.append(s)
#             s = []
#             j = 0
#     # If the last sentence is of length less than sen_length
#     if s != []:
#         sentences.append(s)
#
#     # Vectorizing Sentences using Sklearn to determine Cosine Similarity between adjacent sentences
#     # tfidf_vectorizer = TfidfVectorizer()
#     # tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
#
#
#     matrix = np.zeros([len(sentences), len(keyphrases)])
#     frequency = np.zeros([len(sentences), len(keyphrases)])
#     NOS = np.zeros(len(keyphrases))
#     for i in xrange(len(keyphrases)):
#         for j in xrange(len(sentences)):
#             count = Count_phrase_in_sentence(keyphrases[i].split(' '), sentences[j])
#             frequency[j, i] = count
#             if count > 0:
#                 NOS[i] += 1
#                 # assert(NOS[i] != 0)
#     matrix = frequency / np.log((len(sentences) + eps) / (NOS + eps))
#
#     score = []
#     for i in range(0, matrix.shape[0] - 2):
#         score.append(cosine_similarity(matrix[i:i + 1], matrix[i + 1:i + 2])[0][0])
#
#     # Plotting Cosine Similarity
#     plot_fig(range(1, matrix.shape[0] - 1), score, 'Lexical Similarity with Sentence Length ' + str(sen_len),
#              fig_no)
#
#     # Implementing Windowdiff measure
#     mean_score = np.mean(score)
#     std_score = np.std(score)
#     # Threshold is defined as Mean Score - Standard Deviation
#     threshold = mean_score - std_score
#     boundary = []
#     for i in range(0, len(score) - 2):
#         # score[0] represents the cosine similarity between sentence 1 and sentence 2, score[1] between 2 and 3 and score[2] between 3 and 4
#         # If depth is greater than threshold, then there will be a dissimilariy between sentence 2 and sentence 3, so we are marking sentence 2 as the boundary
#         depth = score[i] - score[i + 1] + score[i + 2] - score[i + 1]
#         if depth >= threshold:
#             boundary.append(i + 1)  # Storing positions of Sentences that represent a boundary
#     # Replacing boundaries with 1 and words with 0
#     segmented_words = []
#     one_segment_words = []
#     for i in range(0, len(sentences)):
#         one_segment_words += sentences[i]
#         if i in boundary:
#             segmented_words.append(one_segment_words)
#             one_segment_words = []
#
#     return segmented_words


#  video part

def candidate_video_words(training_data_path):
    ppt_time = []
    words_pos = []
    ori_words = []
    words_height = []
    f_obj = open(training_data_path,'r')
    lines = f_obj.readlines()
    num_slide = int(lines[0])
    idx = 1
    stemmer = porter.PorterStemmer()
    pos = 0
    for i in xrange(num_slide):
        slide_words_count = 0
        slide_starttime = int(lines[idx])
        ori_words_1slide = []
        words_pos_1slide = []
        words_height_1slide = []
        line = []
        for item in lines[idx + 1].split(','):
            try:
                item = item.encode('ascii', 'ignore').lower()
            except UnicodeDecodeError, e:
                continue
            item = item.split('&')
            if item[0] == '.':
                line.append('.')
                words_height_1slide.append(1e8)
                text = ' '.join(line)
                #print '\n'
                #print text.lower()
                count = coref_pos_filter(text.lower(), words_pos_1slide, ori_words_1slide)
                slide_words_count += count
                #print '\n'
                #print count, len(line)
                assert(count == len(line))
                line = []
            else:
                line.append(item[0].replace(' ','').replace('-','').replace("'",'').replace('cannot','cannt')) # hacking the bug
                words_height_1slide.append(float(item[1]))
        idx += 2
        ori_words.append(ori_words_1slide)
        words_pos.append(words_pos_1slide)
        words_height.append(words_height_1slide)

        #print pos,pos+slide_words_count
        #print ori_words[i]
        #print [val for sublist in ori_words for val in sublist][pos:pos + slide_words_count]
        if len(ppts_time) > 0:
            ppts_time[-1]['endtime'] = slide_starttime
        ppts_time.append({'pos':pos, 'len': slide_words_count, 'starttime':slide_starttime})
        pos += slide_words_count
    ppts_time[-1]['endtime'] = 1e9

    return words_pos, ori_words, words_height


def candidate_phrase(modality, phrases, words_pos, ori_words, words_height, PH, phrase_count):
    phrases_1slide = []

    windowLength = 4
    for i in range(len(ori_words)):
        for j in range(windowLength):
            if i + j  >= len(ori_words):
                break

            # filtering begin
            # rule 1 : the first OR the last word must not be stopword
            if ori_words[i] in stopwords or words_pos[i] not in (POS_TAG['ADJ'] + POS_TAG['NOUN']):
                break
            if ori_words[i + j] in stopwords:
                continue
            # rule 2 : the last word must be noun  AND the consecutive word of the last word must not be noun
            if i + j + 1 < len(ori_words):
                if (words_pos[i + j] not in POS_TAG['NOUN']) or (words_pos[i + j + 1] in POS_TAG['NOUN']):
                    continue
            else:
                if (words_pos[i + j] not in POS_TAG['NOUN']):
                    continue
            # rule 3
            b_skip = False
            for k in xrange(i,i + j + 1):
                #if words_pos[k] in POS_TAG['NOT_NP']:
                if words_pos[k] not in (POS_TAG['ADJ'] + POS_TAG['NOUN'] + POS_TAG['IN'] + POS_TAG['DT']) or len(ori_words[k]) == 1 or isRepetitive(ori_words[k]):
                    b_skip = True
                    break
            if b_skip :
                break
            # rule 4 : phrase can't contain '.' which stands for symbol or delimiter
            b_skip = False
            for k in xrange(i,i + j + 1):
                if ori_words[k] == '.':
                    b_skip = True
                    break
            if b_skip :
                break
            # filtering end

            stem_p = []
            for w in ori_words[i:i + j + 1]:
                stem_p.append(stem_map[w])
            stem_p = ' '.join(stem_p)

            if stem_p not in phrases:
                phrases[stem_p] = ori_words[i:i + j + 1]

            phrase_count[stem_p] = phrase_count.get(stem_p, 0.0) + 1.0

            if modality == 'video':
                PH[stem_p] = max(min(words_height[i:i + j + 1]), PH.get(stem_p,0))
                phrases_1slide.append(stem_p)

    if modality == 'video':
        return list(set(phrases_1slide))

def isRepetitive(word):
    len_word = len(word)
    for i in xrange(len_word):
        if i+2 < len_word:
            if word[i] == word[i+1] == word[i+2]:
                return True
    return False

def Get_PH(PH):
    min_PH = min(PH.values())
    for k,v in PH.iteritems():
        PH[k] = v/min_PH
        #print '%s : %f' % (k, PH[k])

    print_feature_topK(PH, 20)
    return PH


def Get_MOR(phrase_count, num_slide):
    MOR = {}
    num_slide = float(num_slide)
    for k,v in phrase_count.iteritems():
        MOR[k] = v/num_slide
        #print 'MOR %s : %f' % (k, MOR[k])

    #print_feature_topK(MOR, 20)
    return MOR


def Get_COR(phrase_count, phrases_slides):
    COR = {}
    num_slide = len(phrases_slides)
    for phrase, count in phrase_count.iteritems():
        CC, max_CC = 0.0, 0.0
        for i in xrange(num_slide):
            if phrase in phrases_slides[i]:
                CC += 1.0
                if CC > max_CC:
                    max_CC = CC
            else:
                CC = 0.0
        COR[phrase] = max_CC/num_slide * math.log(phrase_count[phrase], 2)

    print_feature_topK(COR, 20)
    return COR

def print_feature_topK(feature, k):
    print "\n\n\n__________________print_feature_topK________________________________"
    test = copy.copy(feature)
    for i in xrange(k):
        t = np.array(test.values())
        idx = np.argmax(t)
        print test.keys()[idx], test[test.keys()[idx]]
        test[test.keys()[idx]] = 0
    print "____________________print_feature_topK______________________________\n\n\n"


def Train_Audio_NB_Classifier():
    phrases_all, ori_words_all, num_word_5min_all, keyphrases_all = [], [], [], []
    ATD_all, Localspan_all, C_Value_all, Cuewords_all, TF_IDF_all = [], [], [], [], []

    for i in xrange(NUM_TRAINING_DATA):
        # #video part
        phrases_t, phrase_count = {}, {}
        training_data_path = "data/train%d" % (i + 1)
        words_pos_t, ori_words_t, num_word_5min_t, sentences_time_t = candidate_audio_words(training_data_path)
        candidate_phrase('audio', phrases_t, words_pos_t, ori_words_t, None, None, phrase_count)
        keyphrases_all.append(read_keyphrases(training_data_path + "_keyphrases", phrases_t))
        phrases_all.append(phrases_t)

        ATD_all.append(Get_ATD(ori_words_t, phrases_t, DEFAULT_WINDOW_LEN))
        Localspan_all.append(Get_Localspan(ori_words_t, phrases_t, num_word_5min_t))
        C_Value_all.append(Get_C_Value(ori_words_t, phrases_t, phrase_count))
        Cuewords_all.append(Get_Cuewords(ori_words_t, phrases_t))

    IDF = Get_IDF(phrases_all)

    for i in xrange(NUM_TRAINING_DATA):
        phrases = phrases_all[i]
        TF_IDF_all.append(Get_TF_IDF(phrases, IDF))

    ATD_all = discretize(ATD_all, keyphrases_all, 'ATD', 'train')
    Localspan_all = discretize(Localspan_all, keyphrases_all, 'Localspan', 'train')
    C_Value_all = discretize(C_Value_all, keyphrases_all, 'C_Value', 'train')
    Cuewords_all = discretize(Cuewords_all, keyphrases_all, 'Cuewords', 'train')
    TF_IDF_all = discretize(TF_IDF_all, keyphrases_all, 'TF_IDF', 'train')

    Audio_NB_Classifier = train_NB_Classifier(
        {'ATD': ATD_all, 'Localspan': Localspan_all, 'C_Value': C_Value_all, 'Cuewords': Cuewords_all,
         'TF_IDF': TF_IDF_all}, keyphrases_all)
    pickle.dump(Audio_NB_Classifier, open('data/OBJ_Audio_NB_Classifier', 'w'))
    pickle.dump(IDF, open('data/OBJ_IDF', 'w'))
    return Audio_NB_Classifier

def Train_Video_NB_Classifier():
    phrases_all, ori_words_all, keyphrases_all, PH_all, MOR_all, COR_all = [], [], [], [], [], []
    for i in xrange(NUM_TRAINING_DATA):
        phrases_t, PH, phrase_count = {}, {}, {}
        phrases_slides = []

        training_data_path = "data/train%d_slide/slide" % (i + 1)
        words_pos_t, ori_words_t, words_height_t, ppt_time_t = candidate_video_words(training_data_path)
        for j in xrange(len(ori_words_t)):
            phrases_1slide = candidate_phrase('video', phrases_t, words_pos_t[j], ori_words_t[j], words_height_t[j], PH, phrase_count)
            phrases_slides.append(phrases_1slide)

        keyphrases_all.append(read_keyphrases("data/train%d_keyphrases" % (i + 1),phrases_t))
        PH_all.append( Get_PH(PH) )
        MOR_all.append( Get_MOR(phrase_count, len(phrases_slides)) )
        COR_all.append( Get_COR(phrase_count, phrases_slides) )

    PH_all = discretize(PH_all, keyphrases_all, 'PH', 'train')
    MOR_all = discretize(MOR_all, keyphrases_all, 'MOR', 'train')
    COR_all = discretize(COR_all, keyphrases_all, 'COR', 'train')

    Video_NB_Classifier = train_NB_Classifier({'PH': PH_all, 'MOR': MOR_all, 'COR': COR_all}, keyphrases_all)
    pickle.dump(Video_NB_Classifier, open('data/OBJ_Video_NB_Classifier', 'w'))

    return Video_NB_Classifier


def Classify_Audio_Keyphrase():
    phrases_t, phrase_count = {}, {}
    test_data_path = "data/test1"
    words_pos_t, ori_words_t, num_word_5min_t = candidate_audio_words(test_data_path)
    candidate_phrase('audio', phrases_t, words_pos_t, ori_words_t, None, None, phrase_count)

    ATD = Get_ATD(ori_words_t, phrases_t, DEFAULT_WINDOW_LEN)
    Localspan = Get_Localspan(ori_words_t, phrases_t, num_word_5min_t)
    C_Value = Get_C_Value(ori_words_t, phrases_t, phrase_count)
    Cuewords = Get_Cuewords(ori_words_t, phrases_t)

    IDF = pickle.load(open('data/OBJ_IDF', 'r'))
    TF_IDF = Get_TF_IDF(phrases_t, IDF)

    ATD_discretized = discretize(ATD, None, 'ATD', 'test')
    Localspan_discretized = discretize(Localspan, None, 'Localspan', 'test')
    C_Value_discretized = discretize(C_Value, None, 'C_Value', 'test')
    Cuewords_discretized = discretize(Cuewords, None, 'Cuewords', 'test')
    TF_IDF_discretized = discretize(TF_IDF, None, 'TF_IDF', 'test')

    Audio_NB_Classifier = pickle.load(open('data/OBJ_Audio_NB_Classifier', 'r'))
    keyphrases, probability = Classify({'ATD': ATD_discretized, 'Localspan': Localspan_discretized, 'C_Value': C_Value_discretized, 'Cuewords': Cuewords_discretized, 'TF_IDF': TF_IDF_discretized},
                          Audio_NB_Classifier)

    return keyphrases, phrases_t, ori_words_t, probability, Cuewords

def Classify_Video_Keyphrase():
    video_NB_Classifier = pickle.load(open('data/OBJ_video_NB_Classifier', 'r'))
    test_data_path = "data/test1"

    words_pos_t, ori_words_t, words_height_t = candidate_video_words(test_data_path + '_slide/slide')
    phrases_t, PH, phrase_count = {},{}, {}
    phrases_slides = []
    for j in xrange(len(ori_words_t)):
        phrases_1slide = candidate_phrase('video', phrases_t, words_pos_t[j], ori_words_t[j], words_height_t[j], PH, phrase_count)
        phrases_slides.append(phrases_1slide)

    MOR = Get_MOR(phrase_count, len(phrases_slides))
    COR = Get_COR(phrase_count, phrases_slides)

    PH_discretized = discretize(PH, None, 'PH', 'test')
    MOR_discretized = discretize(MOR, None, 'MOR', 'test')
    COR_discretized = discretize(COR, None, 'COR', 'test')

    keyphrases, probability = Classify({'PH':PH_discretized, 'MOR':MOR_discretized, 'COR':COR_discretized}, video_NB_Classifier)

    return keyphrases, phrases_t, [y for x in ori_words_t for y in x], probability, PH


def correlation(phrase, audio_ori_words, video_ori_words):
    audio_occur_time = []
    video_occur_time = []
    count = 0
    for st in sentences_time:
        if phrase in audio_ori_words[st['pos']:st['pos']+st['len']]:
            audio_occur_time.append((st['starttime'],st['endtime']))
    for pt in ppts_time:
        if phrase in video_ori_words[pt['pos']:pt['pos']+pt['len']]:
            #video_occur_time.append((pt['starttime'],pt['endtime']))
            for aot in audio_occur_time:
                if ( pt['starttime'] < aot[0] < pt['endtime'] )  or (  pt['starttime'] < aot[1] < pt['endtime']  ):
                    count += 1
                    return count

    return count

def refine_keyphrase(audio_keyphrases, audio_phrases, audio_ori_words, audio_probability, video_keyphrases, video_phrases, video_ori_words, video_probability, PH, Cuewords):
    keyphrases = []
    audio_keyphrases_set = set(audio_keyphrases)
    video_keyphrases_set = set(video_keyphrases)
    audio_phrases_set = set(audio_phrases)
    video_phrases_set = set(video_phrases)

    # rule 1
    keyphrases = keyphrases + list( audio_keyphrases_set.intersection( video_keyphrases_set ) )

    for p in audio_keyphrases_set.intersection( video_phrases_set.difference( video_keyphrases_set ) ):
        if correlation(p,audio_ori_words,video_ori_words) > 0  or PH[p] > 2.5 or Cuewords[p] > 0:
            keyphrases.append(p)

    for p in audio_keyphrases_set.intersection( video_phrases_set ):
        if audio_probability[p] > 0.7: # ???
            keyphrases.append(p)

    for p in video_keyphrases_set.intersection( audio_phrases_set.difference( audio_keyphrases_set ) ):
        if correlation(p,audio_ori_words,video_ori_words) > 0  or PH[p] > 2.5 or Cuewords[p] > 0:
            keyphrases.append(p)

    for p in video_keyphrases_set.intersection( audio_phrases_set ):
        if video_probability[p] > 0.7: # ???
            keyphrases.append(p)

    return keyphrases

def Get_timestamp(time):
    return '%02d:%02d:%02d,%03d' % (time/(60*60000),(time%(60*60000))/60000,(time%(60000))/1000,time%1000)

if __name__ == '__main__':
    NLP_SERVER = StanfordCoreNLP('http://localhost:9000')
    read_stopwords()
    mode = 'test'
    if mode == 'train':
        Audio_NB_Classifier = Train_Audio_NB_Classifier()
        Video_NB_Classifier = Train_Video_NB_Classifier()

    else:
        audio_keyphrases, audio_phrases, audio_ori_words, audio_probability, Cuewords = Classify_Audio_Keyphrase()
        print audio_keyphrases
        print '_______________________________________________________________________________________'
        video_keyphrases, video_phrases, video_ori_words, video_probability, PH = Classify_Video_Keyphrase()
        print video_keyphrases

        keyphrases = refine_keyphrase(audio_keyphrases, audio_phrases, audio_ori_words, audio_probability, video_keyphrases, video_phrases, video_ori_words, video_probability, PH, Cuewords)
        print keyphrases

        segments = text_tiling_slidingwindow(audio_ori_words, keyphrases, window_size=500, sliding_interval=50)
        # for seg in segments:
        #     print len(seg)
        #     print ' '.join(seg),'\n\n\n'

        aligned_segments_time = align_segments_time()
        for i in xrange(len(aligned_segments_time)):
            print Get_timestamp(int(aligned_segments_time[i]['starttime'])), ' --> ', Get_timestamp(int(aligned_segments_time[i]['endtime']))
            print aligned_segments_time[i]['len']
            print ' '.join(audio_ori_words[int(aligned_segments_time[i]['pos']):int(aligned_segments_time[i]['pos']+aligned_segments_time[i]['len'])]),'\n\n\n'

