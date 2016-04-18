from collections import defaultdict
from collections import Counter
import csv
from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups
import re
import random
import numpy as np

democrats = [];
with open('./dem_tweets.txt', 'r') as f:
    democrats = f.read().splitlines();
    
republicans = [];
with open('./rep_tweets.txt', 'r') as f:
    republicans = f.read().splitlines();
    
random.shuffle(republicans)
republicans = republicans[0:len(democrats)]

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=4000,
                                stop_words='english')
def removeHTTPS(l):
    return map(lambda x: re.sub(r"(?:\@|https?\://)\S+", "", x), l)

def removeRT(l):
    return map(lambda x: re.sub(r"(RT )", "", x), l)

democrats = removeHTTPS(democrats)
democrats = removeRT(democrats)

republicans = removeHTTPS(republicans)
republicans = removeRT(republicans)


hashtags_dem_count = {};
hashtags_dem_text = {};
hashtags_rep_count = {};
hashtags_rep_text = {};

for tweet in democrats:
    h = re.findall(r"#(\w+)", tweet);
    h = map(lambda x:x.lower(),h);
    for i in h:
        if hashtags_dem_count.get(i) is None:
            hashtags_dem_count[i] = 1;
            hashtags_dem_text[i] = [tweet];
        else:
            hashtags_dem_count[i] += 1;
            hashtags_dem_text[i].append(tweet);
            
for tweet in republicans:
    h = re.findall(r"#(\w+)", tweet);
    h = map(lambda x:x.lower(),h);
    for i in h:
        if hashtags_rep_count.get(i) is None:
            hashtags_rep_count[i] = 1;
            hashtags_rep_text[i] = [tweet];
        else:
            hashtags_rep_count[i] += 1;
            hashtags_rep_text[i].append(tweet);
            
demHashtagCount = Counter(hashtags_dem_count);
repHashtagCount = Counter(hashtags_rep_count);

print("Democrats:")
for k, v in demHashtagCount.most_common(10):
    print('%s: %i' % (k, v))
    
print("")
print("Republicans:")
for k, v in repHashtagCount.most_common(10):
    print('%s: %i' % (k, v))


num_hashtags = 20;

d = [ seq[0] for seq in demHashtagCount.most_common(500) ]
r = [ seq[0] for seq in repHashtagCount.most_common(500) ]

intersection = list(set(d) & set(r));
popular_hashtags = {};
for i in intersection:
    popular_hashtags[i] = hashtags_dem_count[i] + hashtags_rep_count[i];

pop_h_counter = Counter(popular_hashtags);
popular_hashtags = pop_h_counter.most_common(num_hashtags);
print("Popular Hashtags:")
for k, v in popular_hashtags:
    print('%s: %i' % (k, v))

# calculations for first hashtag
hashtag = popular_hashtags[0][0];
dem_tweets0 = hashtags_dem_text[hashtag];
rep_tweets0 = hashtags_rep_text[hashtag];
num_dem = len(dem_tweets0);
num_rep = len(rep_tweets0);

# compute score of unigram u towards affiliation A
p_dem = len(dem_tweets0)*1.0/(len(rep_tweets0) + len(dem_tweets0));
p_rep = len(rep_tweets0)*1.0/(len(rep_tweets0) + len(dem_tweets0));

count_vec = CountVectorizer(max_df=0.8,  stop_words='english', min_df = 0.0001)

tweets_v = count_vec.fit_transform(dem_tweets0 + rep_tweets0).toarray();

terms = count_vec.get_feature_names();

term_dict = {}
for i in range(len(terms)):
    term_dict[terms[i]] = i
    
index_to_vocab = {i:v for i, v in enumerate(count_vec.get_feature_names())}

dem_tweets_v = tweets_v[0:num_dem,:];
rep_tweets_v = tweets_v[num_dem:,:];
sd = p_dem*np.log(np.divide(np.sum(dem_tweets_v, axis = 0),np.sum(rep_tweets_v, axis = 0)));
sr = p_rep*np.log(np.divide(np.sum(rep_tweets_v, axis = 0), np.sum(dem_tweets_v, axis = 0)));

import math

u_score = {};
u_score_label = {};
for i in range(0, len(sd)):
    sdvalid = not math.isnan(sd[i]) and not math.isinf(sd[i]) and sd[i] !=0;
    srvalid = not math.isnan(sr[i]) and not math.isinf(sr[i]) and sr[i] !=0;
    if (sdvalid and srvalid):
        if (sd[i] > sr[i]):
            u_score[index_to_vocab[i]] = sd[i];
            u_score_label[index_to_vocab[i]] = "d";
        else:
            u_score[index_to_vocab[i]] = sr[i];
            u_score_label[index_to_vocab[i]] = "r";
    elif (sdvalid):
        u_score[index_to_vocab[i]] = sd[i];
        u_score_label[index_to_vocab[i]] = "d";
    elif (srvalid):
        u_score[index_to_vocab[i]] = sr[i];
        u_score_label[index_to_vocab[i]] = "r";


mpscore_dem = [];
mpscore_rep = [];

for i in range(num_dem):
    mpmax = float("-inf");
    text = dem_tweets_v[i,:];
    notzero = np.nonzero(text)[0];
    for j in notzero:
        if (u_score.get(index_to_vocab[j]) is not None):
            if (mpmax < u_score[index_to_vocab[j]]):
                mpmax = u_score[index_to_vocab[j]]
    if(math.isinf(mpmax)):
        mpmax = float("inf");
    mpscore_dem.append(mpmax);

for i in range(num_rep):
    mpmax = float("-inf");
    text = rep_tweets_v[i,:];
    notzero = np.nonzero(text)[0];
    for j in notzero:
        if (u_score.get(index_to_vocab[j]) is not None):
            if (mpmax < u_score[index_to_vocab[j]]):
                mpmax = u_score[index_to_vocab[j]]
    if(math.isinf(mpmax)):
        mpmax = float("inf");
    mpscore_rep.append(mpmax);


all_mp_scores = np.array(mpscore_dem + mpscore_rep);
all_tweets = dem_tweets0 + rep_tweets0;
neutral_tweets = all_mp_scores.argsort()[:10];
for i in neutral_tweets:
    print(all_tweets[i])
    print(all_mp_scores[i])

for i in range(len(all_mp_scores)):
    if (math.isinf(all_mp_scores[i])):
        all_mp_scores[i] = float("-inf");
not_neutral_tweets = all_mp_scores.argsort()[::-1][:10]
for i in not_neutral_tweets:
    print(all_tweets[i])
    print(all_mp_scores[i])