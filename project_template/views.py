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
from django.http import JsonResponse

try:
    from urlparse import urlsplit
except ImportError:
    from urllib.parse import urlsplit
from pymongo import MongoClient

url = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/cs4300')
parsed_url = urlsplit(url)
db_name = parsed_url.path[1:]
print(db_name)

client = MongoClient(url)
db = client[db_name]

if '@' in url:
    user, password = parsed_url.netloc.split('@')[0].split(':')
    db.authenticate(user, password)
    print(user, password)

classifier_tweets = db['unigram_classifier_tweets']

events = db['unigram_classifier_meta_event']

<<<<<<< HEAD
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
=======
event_popularity = db['unigram_classifier_meta_event_popularity']

def get_event_hint(query, n):
    return list(event_popularity.find({"event": re.compile(query, re.IGNORECASE)}, {'event':1, 'popularity':1, '_id':0}).sort('popularity', -1).limit(n))

def get_top_events(n):
    #sample result [{u'popularity': 611.0, u'event': u'GOP'}, {u'popularity': 614.0, u'event': u'tcot'}]
    return list(event_popularity.find({}, {'event':1, 'popularity':1, '_id':0}).sort('popularity', -1).limit(n))

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
>>>>>>> c9ae95d5f05cc239de6327a61927dd6aeaa14f26

# Create your views here.
def index(request):
    dems = ""
    reps = ""
    neutral = ""
    search = ""
    if request.GET.get('search'):
        search = request.GET.get('search')
        dems = classifier_tweets.find({'event': search}).sort("scores.democrats", -1)
        dems = list(dems)[:10]
        reps = classifier_tweets.find({'event': search}).sort("scores.republicans", -1)
        reps = list(reps)[:10]
        neutral = classifier_tweets.find({'event': search})
        neutral = list(sorted(neutral, key=lambda x:max(x["scores"]["democrats"], x["scores"]["republicans"])))[:10]

        dems = word_color(dems, "democrats")
        reps = word_color(reps, "republicans")
        neutral = word_color(neutral, "neutral")
        
        page = request.GET.get('page')
    return render_to_response('project_template/index.html',
                          {'dems': dems,
                           'reps': reps,
                           'neutral': neutral,
                           'search': search,
                           'magic_url': request.get_full_path(),
                           })
