from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from .models import Docs
from django.template import loader
from .form import QueryForm
from .test import find_similar
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

import os
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

# Create your views here.
def index(request):
    output_list = ''
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

        #output_list = find_similar(search)
        #paginator = Paginator(output_list, 10)
        # print(output_list[:10])
        page = request.GET.get('page')
    return render_to_response('project_template/index.html',
                          {'dems': dems,
                           'reps': reps,
                           'neutral': neutral,
                           'search': search,
                           'magic_url': request.get_full_path(),
                           })
