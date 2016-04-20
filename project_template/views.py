from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from .models import Docs
from django.template import loader
from .form import QueryForm
from .test import find_similar
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

import os
from urlparse import urlsplit
from pymongo import MongoClient

url = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/cs4300')
parsed_url = urlsplit(url)
db_name = parsed_url.path[1:]
print db_name

client = MongoClient(url)
db = client[db_name]

if '@' in url:
    user, password = parsed_url.netloc.split('@')[0].split(':')
    db.authenticate(user, password)
    print user, password

classifier_tweets = db['unigram_classifier_tweets']

# Create your views here.
def index(request):
    output_list = ''
    output=''
    if request.GET.get('search'):
        search = request.GET.get('search')
        output_list = list(classifier_tweets.find({'event': search}))
        #output_list = find_similar(search)
        paginator = Paginator(output_list, 10)
        page = request.GET.get('page')
        try:
            output = paginator.page(page)
        except PageNotAnInteger:
            output = paginator.page(1)
        except EmptyPage:
            output = paginator.page(paginator.num_pages)
    return render_to_response('project_template/index.html',
                          {'output': output,
                           'magic_url': request.get_full_path(),
                           })
