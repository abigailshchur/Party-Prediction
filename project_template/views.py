from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from .models import Docs
from django.template import loader
from .form import QueryForm
from .test import find_similar
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

import re
import os
from collections import defaultdict
from django.http import JsonResponse
try:
    from urlparse import urlsplit
except ImportError:
    from urllib.parse import urlsplit
#from nltk.sentiment.vader import SentimentIntensityAnalyzer

from pymongo import MongoClient
import tweepy

import sys
# sys.path.insert(0, '../jobs')
# sys.path.append('../jobs/')
job_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'jobs')
sys.path.append(job_dir)
#import scrapper

#from project_template import classifier

url = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/cs4300')
parsed_url = urlsplit(url)
db_name = parsed_url.path[1:]
#print(db_name)

client = MongoClient(url)
db = client[db_name]

if '@' in url:
    user, password = parsed_url.netloc.split('@')[0].split(':')
    db.authenticate(user, password)
    print(user, password)

classifier_tweets = db['unigram_classifier_tweets']
events = db['unigram_classifier_meta_event']
event_popularity = db['unigram_classifier_meta_event_popularity']
classifier_terms = db['unigram_classifier_meta_term']
unigram_scores = db['unigram_scores']
#event_scores = db['event_scores']
#unigram_scores = db['unigram_scores']

#print(urllib.request.urlopen("http://www.sentiment140.com/api/classify?text=new+moon+is+awesome&query=new+moon").read())

def get_event_hint(query, n):
    return list(event_popularity.find({"event": re.compile(query, re.IGNORECASE)}, {'event':1, '_id':0}).sort('avg', -1).limit(n))

def format_suggestion(query, l):
    r = {};
    r['query'] = query;
    r['suggestions'] = [{'value':x['event'], 'data':x['event']} for x in l]
    return r;

def search_hint(request):
    #url example /pt/search_hint?query=[...]
    n = 10
    if request.GET.get('query'):
        query = request.GET.get('query')
        return JsonResponse(format_suggestion(query, get_event_hint(query, n)), safe=False)
    else:
        return JsonResponse(format_suggestion('', get_top_events(n)), safe=False)

def get_top_events(n):
    #sample result [{"republicans" : 875, "avg" : 674, "democrats" : 473, "event" : "gop" }]
    items = list(event_popularity.find({}).sort('avg', -1).limit(n))
    for item in items:
        item['popularity'] = item['avg']
        if 'democrats' not in item:
            item['democrats'] = 0
        if 'republicans' not in item:
            item['republicans'] = 0
        num_dems = item['democrats']
        num_reps = item['republicans']
        total = num_dems + num_reps
        red = int(float(num_reps) / total * 255)
        blue = int(float(num_dems) / total * 255)
        item["color"] = "rgb(" + str(red) + ",0," + str(blue) + ")"
    print(items[0])
    return items

def word_color(x, side_or_neutral):
    for i in range(len(x)):
        line = x[i]
        words = line["tweet"]["text"].split()
        x[i]["words"] = []
        for word in words:
            dic = {}
            dic["text"] = word.replace("&amp;", "&")
            red_alpha = 0
            blue_alpha = 0
            sides = ["democrats", "republicans"] if side_or_neutral == "neutral" else [side_or_neutral]
            for side in sides:
                score_detail = dict(line["score_detail"][side])
                if (word.startswith("#")):
                    word = word[1:]
                if word in score_detail:
                    #print(score_detail[word])
                    dic["highlighted"] = True
                    if (score_detail[word] > 0 and side == "democrats") or (score_detail[word] < 0 and side == "republicans"):
                        blue_alpha = pow(abs(score_detail[word]), 0.3)
                        #dic["color"] = "rgba(0, 0, 255, " +  + ")"
                    else:
                        red_alpha = pow(abs(score_detail[word]), 0.3)
                        #dic["color"] = "rgba(255, 0, 0, " +  + ")"
                else:
                    dic["highlighted"] = False
            red = int(255 * red_alpha)
            blue = int(255 * blue_alpha)
            alpha = red_alpha + blue_alpha*(1-red_alpha)
            dic["text_color"] = "#000" if alpha < 0.5 else "#ccc"
            dic["color"] = "rgba(" + str(red) + ",0," + str(blue) + "," + str(alpha) + ")"
            x[i]["words"].append(dic)
    return x

def distinct(l, key):
    r = []
    keys_tokens = []
    for x in l:
        text = key(x).lower().strip()
        tokens = text.split()
        nonhashlen = len(tokens)-len([i[1:] for i in text.split() if i.startswith("#") or i.startswith("http://") or i.startswith("https://")])
        textlen = len([i[1:] for i in text.split()])

        # repeat if more than 4 words in common
        repeat = False
        for toks in keys_tokens:
            if len(toks.intersection(set(tokens))) > 4:
                repeat = True
                break
        if not repeat and nonhashlen > 1 and textlen > 5:
            r.append(x)
            keys_tokens.append(set(tokens))
    return r

#def mostly_neu(d):
#    return d["neu"] > d["pos"] and d["neu"] > d["neg"]

def calculate_score(terms, event):
    scores = defaultdict(list)
    for t in terms:
        for affiliation in ["democrats", "republicans"]:
            uni = unigram_scores.find_one({"event": event, "affiliation": affiliation, "term": t})
            new_score = uni["score"] if uni else 0
            scores[affiliation].append(new_score)
    return {k:max(v) for k,v in scores.items()}, {k:list(zip(terms, v)) for k,v in scores.items()}

## Return democrat score and republican score tuple ##
def score_tweet(tweet, event):
    tweet_dict = {}
    
    ## Use this for classification ##
    word_array = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet['text']).split()
    terms = [x.lower() for x in word_array]
    
    tweet_dict['id'] = tweet['id']
    ## Rest is formatting ##
    #user = tweet['user']
    tweet_dict['tweet']={'text': tweet['text'], 'user_id': tweet['user'].id, 'created_at' : tweet['created_at']}
    affil_scores, term_scores = calculate_score(terms, event)
    tweet_dict['scores'] = affil_scores #{'democrats':5, 'republicans':5}
    tweet_dict['score_detail'] = term_scores
    #print(tweet_dict)
    #tweet_dict['score_detail'] = {'democrats': [], 'republicans': []}
    #for i in word_array:
    #    tweet_dict['score_detail']['democrats'].append([i, 2])
    #    tweet_dict['score_detail']['republicans'].append([i, 2])
    return tweet_dict


def get_to_tweets(event):
    #sid = SentimentIntensityAnalyzer()
    new_tweets = get_tweets_for_a_hashtag(event, num_tweets = 70, views = ['text', 'id', 'user', 'created_at'])
    tweets = list(classifier_tweets.find({'event': event}))
    for i in new_tweets:
        tweets.append(score_tweet(i, event))
    #t = scrapper.classifier
    #calculate_score("test test", event)
    tweets = distinct(tweets, lambda x: x['tweet']['text'])
    dems = list(sorted(tweets, key=lambda x: x["scores"]["democrats"],   reverse=True))
    reps = list(sorted(tweets, key=lambda x: x["scores"]["republicans"], reverse=True))

    #neu_classes = classifier.predict(tweets)
    #neutral = [x for i, x in enumerate(tweets) if neu_classes[i] == 1]
    neutral = list(sorted(tweets, key=lambda x:max(x["scores"]["democrats"], x["scores"]["republicans"])))

    #print("#### DEMS ######")
    #print(classifier.predict(dems))
    #print("#### REPS ######")
    #print(classifier.predict(reps))
    #print("#### NEUTRAL ######")
    #print(classifier.predict(neutral))
    return (dems[:10], reps[:10], neutral[:10])

username_cache = {}

def insert_usernames(api, ls):
    for i in range(len(ls)):
        user_id = ls[i]["tweet"]["user_id"]
        if user_id in username_cache:
            ls[i]["username"] = username_cache[user_id]
        else:
            try:
                u = api.get_user(user_id)
                ls[i]["username"] = u.screen_name
                username_cache[user_id] = u.screen_name
            except:
                ls[i]["username"] = ""
    return ls

keys = [["BDzpFtaKFVwFi" + "M5Xj0uSgu0hF", "M8WoPc1DUfXaAyUOGSFSP4G87LDNe192QY7G" + "Mbie8lpPqPxwK6", "717950588076601344-k7gPVk" + "dbDSP0aktBF1tNSIFnpsu5XI3", "R2Fc8zG3U0DwpDnX5MqftlVbmR" + "zQK5HSlIrII29U8wMFM"], ["eNfjPJT12a1aiFGaVSNnn6nTg", "wJk3RhuhUo5MFNnnLaJQIM2Q" + "93gFeMMfWGUzoYd6z49z8Kis2w", "717950588076601344-yDbU6iN" + "96hMagodDyv2iqTxuiNQ7VkS", "OycOAMytzLXlik4qO3" + "2iLWxPoPaqNmoXlDrW6QfhhX7Vd"], ["Qvp9YhLDqwngJcX" + "ixh95xG6U2", "I6ZfZHsAaHbmtGuYfo9Ku8G" + "ngZ86I2X9rQob4e9imHcQjRLd0C", "717950588076601344-kJkuGPJZIT" + "NQnZsFGUpHr9Ru76k60bu", "RHKloyH0d7FjeYUC" + "FJB4m36cXlh8hA6b9QJOFsugGhzTy"]]
cur_key = 0

# Create your views here.
def index(request):
    global cur_key
    dems = ""
    reps = ""
    neutral = ""
    search = ""
    if request.GET.get('search'):
        search = request.GET.get('search')
        dems, reps, neutral = get_to_tweets(search)

        dems = word_color(dems, "democrats")
        reps = word_color(reps, "republicans")
        neutral = word_color(neutral, "neutral")

        auth = tweepy.OAuthHandler(keys[cur_key][0], keys[cur_key][1])
        auth.set_access_token(keys[cur_key][2], keys[cur_key][3])
        api = tweepy.API(auth)
        cur_key += 1
        if cur_key >= len(keys):
            cur_key = 0
        
        dems = insert_usernames(api, dems)
        reps = insert_usernames(api, reps)
        neutral = insert_usernames(api, neutral)
        
        page = request.GET.get('page')
    return render_to_response('project_template/index.html',
                              {'dems': dems,
                               'reps': reps,
                               'neutral': neutral,
                               'search': search,
                               'top_events': get_top_events(30),
                               'magic_url': request.get_full_path(),
                              })

def get_tweets_for_a_hashtag(hashtag, num_tweets = 10, views = ['text']):
    global cur_key
    auth = tweepy.OAuthHandler(keys[cur_key][0], keys[cur_key][1])
    auth.set_access_token(keys[cur_key][2], keys[cur_key][3])
    api = tweepy.API(auth)
    cur_key += 1
    if cur_key >= len(keys):
        cur_key = 0
    qr = tweepy.Cursor(api.search, q=hashtag, lang="en").items(num_tweets)
    out_list = []
    for twit in qr:
        twit_dickt = twit.__dict__
        out_dickt = {}
        for view in views:
            try:
                out_dickt[view] = twit_dickt[view]
            except:
                raise
        out_list.append(out_dickt)
    return out_list
