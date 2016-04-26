#!/usr/bin/python

import time
import datetime
import tweepy
import csv
from pymongo import MongoClient
import pymongo
import sys
import math
import re
import os
from collections import defaultdict
try:
    from urlparse import urlsplit
except ImportError:
    from urllib.parse import urlsplit
from bson.binary import Binary
import gzip
try:
    import cPickle as pickle
except ImportError:
    import pickle
import json
import io
import base64
import random

os.chdir(os.path.dirname(os.path.realpath(__file__)))
import id_list

stop_words = set([u'all', 'gt', 'forget', 'ron', 'global', 'month', 'steve', 'talks', 'follow', 'chair', 'children', 'issues', 'increase', 'votes', 'voter', 'tx', 'tv', 'tweet', 'obamacare', 'young', u'to', 'program', 'voted', u'under', 'must', 'veterans', 'brown', u'very', 'continues', 'choice', 'facebook', u'yourselves', 'ht', 'school', 'hour', u'did', 'list', 'democrat', 'ryan', 'race', 'team', 'small', 'radio', 'bernie', "i'll", 'mn', 'says', 'leaders', 'excited', 'tea', 'ted', 'sign', 'dnc', 'go', 'rate', 'htt', 'video', 'pass', u'further', 'air', 'even', u'what', 'power', 'anniversary', 'liberty', 'find', 'access', u'above', 'new', 'ever', 'public', 'dem', 'supporting', 'full', 'iran', 'healthcare', 'join', 'men', u'here', 'hours', 'let', 'address', u'hers', '100', 'strong', 'change', 'wait', 'great', 'kids', 'potus', '30', 'social', 'action', 'military', "i'm", 'makes', 'honor', 'via', 'love', 'family', 'win', 'county', 'put', 'patriots', '1st', 'crisis', 'standing', 'discuss', 'use', u'from', 'takes', 'usa', 'visit', 'two', 'next', u'few', 'live', 'call', '6', 'candidates', u'themselves', u'until', 'pres', u'more', 'yes', 'supporters', 'share', 'freedom', 'hold', 'glad', 'women', u'me', 'high', 'word', 'mc', 'rights', u'this', 'work', 'mi', 'mt', u'can', u'of', 'meet', u'my', 'history', 'control', 'trump', 'give', "didn't", 'states', 'heard', 'minimum', 'something', 'want', 'times', 'needs', 'united', 'court', 'protect', 'breaking', '1', u'how', 'interview', 'economy', 'plans', 'may', 'stop', u'after', 'water', 'wrong', 'birthday', 'coming', u'such', 'law', 'fl', 'response', 'man', u'a', 'remember', 'save', 'st', u'so', 'media', 'every', 'democratic', 'talk', "that's", 'bush', 'help', 'office', "don't", u'over', 'move', 'trade', u'through', 'unite', 'committee', 'cuts', 'still', u'its', u'before', 'police', 'thank', "he's", 'fix', 'better', '2015', u'ours', 'msnbc', 'bipartisan', 'petition', '2012', u'then', u'them', 'good', 'iowa', 'food', 'dan', 'safe', "obama's", u'they', u'not', u'now', 'hall', u'nor', 'equality', 'name', 'always', '--', 'boehner', 'victory', 'truth', u'each', 'done', 'bills', "isn't", 'everyone', 'covered', u'doing', 'ed', 'energy', 'hard', 'rally', 'joined', 'year', u'our', 'calling', 'saturday', 'special', u'out', u'rt', 'perry', 'funding', 'won', 'rs', 'miss', 'rep', 'since', 'looking', 're', 'health', 'hill', '7', 'got', 'get', 'gov', 'issue', 'gop', 'red', 'shows', 'tcot', "you're", 'friday', 'little', 'free', '20', 'members', 'ask', 'care', 'benefits', 'district', 'discussing', 'could', 'florida', 'days', 'david', 'thing', 'american', 'place', "today's", 'cain', 'polls', 'think', 'south', 'first', 'already', 'city', 'xl', 'feel', 'legislation', 'primary', u'yourself', 'americans', 'another', 'president', 'vote', 'message', 'fight', 'open', 'tomorrow', 'millions', "doesn't", 'story', 'service', 'together', 'top', 'system', u'their', 'attack', '2', u'too', 'statement', 'tom', 'white', 'john', 'final', 'friend', 'listen', 'passed', u'that', 'convention', 'hillary', 'part', 'believe', u'herself', u'than', "here's", 'obama', '11', 'king', '13', 'b', '15', '14', '16', 'project', 'matter', 'cost', 'marriage', 'speaking', u'were', 'sec', u'and', 'tonight', 'talking', 'say', u'have', 'need', u'any', 'congrats', 'congressional', '-', 'also', 'without', 'take', u'which', 'online', 'blue', 'sure', u'who', u'most', 'plan', 'letter', 'sb', 'nothing', 'america', u'why', 'million', 'volunteers', u'don', 'clear', 'proud', 'm', 'face', 'clean', 'latest', 'fact', 'joining', 'show', 'bring', 'businesses', 'democrats', '2013', 'debate', 'staff', 'one', 'o', 'justice', 'true', 'unemployment', 'enough', "won't", u'should', u'only', 'going', 'black', 'rick', 'photo', 'folks', '8', 'watching', 'local', 'hope', u'do', u'his', 'means', 'michigan', 'de', 'watch', 'dc', '12', 'report', u'during', 'years', 'dr', u'him', 'celebrate', 'h', 'taxes', 'morning', 'reid', 'bad', u'she', 'ohio', u'where', 'learn', 'fair', 'govt', 'national', 'see', 'college', u'are', 'sen', 'retweet', 'best', '03', 'said', 'reform', 'federal', 'away', 'please', 'mayor', '3', 'future', u'between', 'progress', 'across', 'caucus', 'affordable', u'we', 'never', 'jobs', 'capitol', 'wi', 'job', 'cut', 'joe', 'group', 'debt', 'come', u'both', 'climate', 'last', 'candidate', 'country', 'taking', 'la', 'equal', u'against', u's', 'senator', "can't", 'insurance', 'co', 'ca', 'community', 'news', 'poll', 'speak', 'conference', 'blog', 'raise', 'union', 'west', 'create', 'political', u'been', 'mark', u'whom', 'pa', 'meeting', 'pm', 'wants', 'sanders', 'life', 'families', 'fire', 'romney', 'republicans', 'c', 'al', 'policy', 'lives', '2014', 'former', u'those', 'case', 'commerce', u'myself', '2016', 'look', u'these', 'bill', 'budget', 'governor', 'n', u'will', 'w', u'while', 'texas', 'policies', 'many', 'voice', 'va', 'would', 'voters', u'is', 'speech', u'it', 'middle', u'itself', 'im', 'dems', u'in', 'ready', u'if', 'il', 'pay', 'make', 'tour', u'same', 'rules', 'speaker', '9', 'party', 'gets', "it's", 'higher', 'week', 'http', 'yesterday', 'running', '000', 'student', 'nation', 'opportunity', 'tune', 'keep', 'ofa', u'off', 'center', u'i', 'floor', 'well', 'cruz', 'fighting', 'costs', u'the', u'yours', 'left', 'things', 'icymi', u'just', 'less', u'being', 'money', 'weekend', 'gun', 'schools', 'thanks', 'questions', 'world', 'paul', 'yet', 'wage', 'republican', 'helping', 'campaign', 'ma', 'majority', 'workers', u'had', 'day', 'barack', 'heritage', 'add', '4', 'thoughts', 'tax', u'has', 'tuesday', 'build', 'real', 'march', 'around', 'government', 'read', 'big', 'early', 'game', 'biz', 'know', 'press', 'amp', 'd', 'like', 'lost', '50', 'continue', u't', 'night', 'security', 'works', 'making', 'amendment', u'because', 'old', 'crowd', 'people', 'senate', 'twitter', u'some', 'back', 'donald', 'growth', 'election', 'home', u'ourselves', u'for', 'decision', 'fox', 'conservative', u'does', 'keystone', 'cnn', 'ct', 'leader', u'be', 'run', 'business', 'mitt', 'fed', 'step', 'leadership', 'chairman', 'post', 'r', 'super', u'by', 'hearing', u'on', u'about', 'working', 'oh', u'theirs', 'violence', 'scott', 'stand', 'act', 'congressman', 'efforts', u'or', u'own', 'clinton', u'into', 'washington', 'getting', u'down', 'right', 'honored', u'your', 'deal', u'her', 'spending', 'friends', 'support', u'there', 'question', 'hey', 'long', 'class', 'start', 'much', "we're", 'way', 'house', 'forward', 'yrs', u'was', 'war', 'happy', 'ago', u'himself', 'hr', 'immigration', u'but', 'state', 'hear', 'line', 'bern', u'with', u'he', 'info', 'congratulations', 'made', 'calls', u'up', 'signed', 'tell', 'record', u'below', 'economic', 'stories', 'today', 'problem', 'minutes', 'agree', 'supreme', 'ad', 'ag', u'am', 'mike', u'an', u'as', 'ar', u'at', 'et', 'politics', 'education', 'check', 'nh', u'again', 'us', u'no', 'deserve', 'nc', u'when', 'event', 'end', 'ny', u'other', '5', u'you', 'really', '10', "'s", 'congress', 'students', 'welcome', 'stay', "let's", 'town', 'important', 'coverage', 'chris', 'repeal', 'pelosi', 'billion', 'voting', 'lead', 'using', 'fund', 'daily', 'u', 'time', 'walker', u'having', u'once'])

#user_idx = {x: i for i, x in enumerate(id_list.democrats + id_list.republicans)}
user_idx = id_list.democrats + id_list.republicans


class Logger(object):
    outs = [sys.stdout]

    def __init__(self, tag):
        self.tag = tag

    def stamp(self, message):
        return '[{}][{}]--{}\n'.format(self.tag, datetime.datetime.now(), message)

    def d(self, message):
        self.write(message)

    def write(self, message):
        for o in Logger.outs:
            o.write(self.stamp(message))

    def flush(self):
        pass

class Unigram_Classifier_DB:
    def __init__(self, url, db_name, affiliations):
        self.log = Logger("Unigram_Classifier_DB")
        self.tweet_filter_range = 50
        self.client = MongoClient(url)
        self.db = self.client[db_name]
        self.classifier_tweets = self.db['unigram_classifier_tweets']
        self.classifier_meta_event = self.db['unigram_classifier_meta_event']
        self.classifier_meta_term = self.db['unigram_classifier_meta_term']
        self.classifier_tweets_cache = defaultdict(list)
        self.classifier_meta_event_cache = {}
        self.classifier_meta_term_cache = {}
        self.classifier_meta_event_cache_count = {}
        self.classifier_meta_term_cache_count = {}
        self.affiliations = affiliations
        self.setup_index()

    def setup_index(self):
        classifier_tweets_index = [('tweet.tweet_id', pymongo.ASCENDING), ('event', pymongo.ASCENDING)]
        for affiliation in self.affiliations:
            classifier_tweets_index.append(('scores.'+affiliation, pymongo.ASCENDING))
        self.classifier_tweets.create_index(classifier_tweets_index)
        self.classifier_meta_event.create_index([('event', pymongo.ASCENDING), ('affiliation', pymongo.ASCENDING)])
        self.classifier_meta_term.create_index([('event', pymongo.ASCENDING), ('term', pymongo.ASCENDING), ('affiliation', pymongo.ASCENDING)])

    def clear(self):
        self.classifier_tweets.drop()
        self.classifier_meta_event.drop()
        self.classifier_meta_term.drop()

    def update_event_inc(self, event, user_id, affiliation):
        if event not in self.classifier_meta_event_cache:
            self.get_event_meta(event)
        if user_id not in self.classifier_meta_event_cache[event][affiliation]:
            self.classifier_meta_event_cache[event][affiliation].add(user_id)
            self.classifier_meta_event_cache_count[event][affiliation] += 1

    def update_term_inc(self, event, term, user_id, affiliation):
        if event not in self.classifier_meta_term_cache:
            self.get_term_meta(event, term)
        if (user_id, term) not in self.classifier_meta_term_cache[event][affiliation]:
            self.classifier_meta_term_cache[event][affiliation].add((user_id, term))
            if (event, term) not in self.classifier_meta_term_cache_count:
                self.classifier_meta_term_cache_count[(event, term)] = {affiliation:0 for affiliation in self.affiliations}
            self.classifier_meta_term_cache_count[(event, term)][affiliation] += 1

    def binary_to_obj(self, binary):
        x = io.BytesIO(binary)
        f = gzip.GzipFile(mode='r', fileobj=x)
        obj = pickle.load(f)
        f.close()
        x.close()
        return obj

    def obj_to_binary(self, obj):
        x = io.BytesIO()
        f = gzip.GzipFile(mode='wb', fileobj=x)
        pickle.dump(obj, f)
        f.close()
        binary = Binary(x.getvalue())
        x.close()
        return binary

    def get_event_meta(self, event):
        #event_cache {event: {affiliation: [user_ids]} } #event_cache_count {event: {affiliation: int}}
        if event in self.classifier_meta_event_cache:
            return self.classifier_meta_event_cache_count[event]
        data = {c['affiliation']:set(c['user_ids']) for c in self.classifier_meta_event.find({'event': event})}
        for affiliation in self.affiliations:
            if affiliation not in data:
                data[affiliation] = set()
        self.classifier_meta_event_cache[event] = data
        self.classifier_meta_event_cache_count[event] = {affiliation:len(user_ids) for affiliation, user_ids in data.items()}
        return self.get_event_meta(event)

    def get_term_meta(self, event, term):
        #event_cache {event: {affiliation: [user_id_term_pairs]} } #event_cache_count {(event,term): {affiliation: int}}
        if event in self.classifier_meta_term_cache:
            if (event,term) in self.classifier_meta_term_cache_count:
                return self.classifier_meta_term_cache_count[(event,term)]
            else:
                return {affiliation:0 for affiliation in self.affiliations}
        data = {c['affiliation']:self.binary_to_obj(c['user_id_term_pairs']) for c in self.classifier_meta_term.find({'event': event})}
        for affiliation in self.affiliations:
            if affiliation not in data:
                data[affiliation] = set()
        self.classifier_meta_term_cache[event] = data
        #setup classifier_meta_term_cache_count
        for affiliation, user_id_term_pairs in data.items():
            for user_id, term in user_id_term_pairs:
                if (event, term) not in self.classifier_meta_term_cache_count:
                    self.classifier_meta_term_cache_count[(event, term)] = {affiliation:0 for affiliation in self.affiliations}
                self.classifier_meta_term_cache_count[(event, term)][affiliation] += 1
        return self.get_term_meta(event, term)

    def store_tweet(self, tweet, event, scores, score_detail):
        assert len(scores) == len(self.affiliations)
        self.classifier_tweets_cache[event].append({'tweet': tweet, 'event': event, 'scores': scores, 'score_detail': score_detail})

    def flush_tweets(self):
        to_add_all = []
        to_remove_all = set()
        for c in list(self.classifier_tweets.find({'event': {'$in':list(self.classifier_tweets_cache.keys())}}, ['scores', 'event'])):
            self.classifier_tweets_cache[c['event']].append(c)
        self.log.d('flush tweets #events {}'.format(len(self.classifier_tweets_cache)))
        for event, tweets in self.classifier_tweets_cache.items():
            to_remove = set([x['_id'] for x in tweets if '_id' in x])
            to_add = {}
            for affiliation in self.affiliations:
                tweets.sort(key = lambda x: x['scores'][affiliation])
                to_remove.intersection_update([x['_id'] for x in tweets[self.tweet_filter_range:-self.tweet_filter_range] if '_id' in x])
                for x in tweets[0:self.tweet_filter_range] + tweets[-self.tweet_filter_range:]:
                    if '_id' not in x:
                        to_add[(x['event'], x['tweet']['tweet_id'])] = x
            to_add_all += list(to_add.values())
            to_remove_all.update(to_remove)
        if to_remove_all:
            self.log.d('remove tweets {}'.format(len(to_remove_all)))
            self.classifier_tweets.remove({'_id':{'$in':list(to_remove_all)}})
        if to_add_all:
            self.log.d('add tweets {}'.format(len(to_add_all)))
            self.classifier_tweets.insert_many(to_add_all)
        self.classifier_tweets_cache.clear()

    def flush_event_supplementary(self):
        pipline = [{'$match'  : {}},
            	   {'$project': {'event': 1, 'affiliation': 1, 'count': { '$size': '$user_ids' }}},
            	   {'$group' : { '_id' : '$event', 'popularity': { '$avg': "$count" }}},
            	   {'$project': {'event': '$_id', 'popularity': 1}},
                   {'$out': 'unigram_classifier_meta_event_popularity'}
	               ]
        self.classifier_meta_event.aggregate(pipline)
        self.log.d('complete flush_event_supplementary')

    def flush(self, insert=False, use_pickle=False):
        self.flush_tweets()
        if (use_pickle):
            self.log.d('flush using pickle')
            with gzip.GzipFile('classifier_meta_event.pgz', 'w+') as f:
                pickle.dump(self.classifier_meta_event_cache, f)
            with gzip.GzipFile('classifier_meta_term.pgz', 'w+') as f:
                pickle.dump(self.classifier_meta_term_cache, f)
            self.classifier_meta_event_cache.clear()
            self.classifier_meta_term_cache.clear()
            return
        self.log.d('flush event_meta {}'.format(len(self.classifier_meta_event_cache)))
        for event, data in self.classifier_meta_event_cache.items():
            for affiliation, user_ids in data.items():
                if user_ids:
                    if insert:
                        self.classifier_meta_event.insert({'event':event, 'affiliation':affiliation, 'user_ids':list(user_ids)})
                    else:
                        self.classifier_meta_event.update_one({'event':event, 'affiliation':affiliation}, {"$set": {'user_ids':  list(user_ids) }}, upsert=True)
        self.log.d('flush term_meta {}'.format(len(self.classifier_meta_term_cache)))
        for event, data in self.classifier_meta_term_cache.items():
            for affiliation, user_id_term_pairs in data.items():
                if user_id_term_pairs:
                    if insert:
                        self.classifier_meta_term.insert({'event':event, 'affiliation':affiliation, 'user_id_term_pairs':self.obj_to_binary(user_id_term_pairs)})
                    else:
                        self.classifier_meta_term.update_one({'event':event, 'affiliation':affiliation}, {"$set": {'user_id_term_pairs':  self.obj_to_binary(user_id_term_pairs) }}, upsert=True)
        self.classifier_meta_event_cache.clear()
        self.classifier_meta_term_cache.clear()
        self.flush_event_supplementary()

    def close(self):
        self.client.close()

class Unigram_Classifier:
    def __init__(self, db):
        self.db = db

    def get_tags(self, tweet):
        return [tag['text'].lower() for tag in tweet['hashtags']]

    def clean_text(self, text):
        return re.sub(r"http\S+", "", text)

    def get_terms(self, tweet):
        text = self.clean_text(tweet['text'])
        terms = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", text)
        terms = [t.lower() for t in terms]
        terms = [t for t in terms if t not in stop_words]
        return set(terms)

    def calculate_score(self, terms, event):
        scores = defaultdict(list)
        event_counts = self.db.get_event_meta(event)
        if any([c < 50 for c in event_counts.values()]):
            return None
        if len(terms) == 0:
            return None
        for t in terms:
            term_counts = self.db.get_term_meta(event, t)
            assert len(term_counts) == 2
            for affiliation, term_count in term_counts.items():
                other_term_count = sum([c for a,c in term_counts.items() if a != affiliation])
                other_event_count = sum([c for a,c in event_counts.items() if a != affiliation])
                other_portion = (float(other_term_count)+1)/(other_event_count+1)
                this_portion = (float(term_count)+1)/(event_counts[affiliation]+1)
                new_score = this_portion * math.log(this_portion / other_portion)
                scores[affiliation].append(new_score)
        return {k:max(v) for k,v in scores.items()}, {k:list(zip(terms, v)) for k,v in scores.items()}

    def learn(self, tweet, affiliation):
        terms = self.get_terms(tweet)
        events = self.get_tags(tweet)
        user_id = tweet['user_id']
        for event in events:
            self.db.update_event_inc(event, user_id, affiliation)
            for term in terms:
                self.db.update_term_inc(event, term, user_id, affiliation)
            scores = self.calculate_score(terms, event)
            if scores is None:
                continue
            scores, score_detail = scores
            self.db.store_tweet(tweet, event, scores, score_detail)

class Scrapper_META_DB:
    def __init__(self, url, db_name):
        self.client = MongoClient(url)
        self.db = self.client[db_name]
        self.scrape_meta_collection = self.db['scraper_meta']
        self.lastest_tweet_id_cache = {}

    def clear(self):
        self.scrape_meta_collection.drop()

    def update_latest_tweet_id_of_user(self, user_id, tweet_id):
        if user_id not in self.lastest_tweet_id_cache:
            get_latest_tweet_id_of_user(user_id)
        if tweet_id > self.lastest_tweet_id_cache[user_id]:
            self.lastest_tweet_id_cache[user_id] = tweet_id

    def get_latest_tweet_id_of_user(self, user_id):
        if user_id in self.lastest_tweet_id_cache:
            if self.lastest_tweet_id_cache[user_id] == -1:
                return None
            else:
                return self.lastest_tweet_id_cache[user_id]
        row = self.scrape_meta_collection.find_one({'user_id':user_id})
        if row is None:
            self.lastest_tweet_id_cache[user_id] = -1
        else:
            tweet_id = row['latest_tweet_id']
            self.lastest_tweet_id_cache[user_id] = int(tweet_id)
        return self.get_latest_tweet_id_of_user(user_id)

    def flush(self):
        for user_id, tweet_id in self.lastest_tweet_id_cache.items():
            if tweet_id == -1:
                continue
            self.scrape_meta_collection.update_one({'user_id':user_id}, {"$set": {'latest_tweet_id': tweet_id}}, upsert=True)

    def close(self):
        self.client.close()

class Scrapper:
    def __init__(self, API_pool, scrapper_meta, callback):
        self.log = Logger("Scrapper")
        self.API_pool = API_pool
        self.scrapper_meta = scrapper_meta
        self.callback = callback

    def scrape(self, target):
        user_id = target['user_id']
        latest_tweet_id = self.scrapper_meta.get_latest_tweet_id_of_user(user_id)
        try:
            tweets = self.API_pool.fetch_tweets(user_id, latest_tweet_id)
        except tweepy.TweepError as e:
            self.log.d('<{}> TweepError: {}'.format(target['user_id'], str(e)))
            return
        except:
            self.log.d('<{}> Unexpected error: {}'.format(target['user_id'], sys.exc_info()[0]))
            raise
        if len(tweets) == 0:
            return
        for tweet in tweets:
            self.callback(tweet, target)
            self.scrapper_meta.update_latest_tweet_id_of_user(user_id, tweet['tweet_id'])

    def run(self, targets):
        total = len(targets)
        for i, target in enumerate(targets):
            user_id = target['user_id']
            self.log.d('scrape {}th user_id: {} {:0.3f}%'.format(i, user_id, float(i)/total*100))
            try:
                self.scrape(target)
            except API_pool.AllBusyError:
                self.log.d('AllBusyError')
                break
        self.log.d('end run')

class API_pool():
    class AllBusyError(Exception):
        pass

    def __init__(self, authentications, wait=False):
        self.wait = wait
        self.log = Logger("API_pool");
        self.APIs = [];
        self.status = []; #-1 is idle, time is busy
        if len(authentications) == 0:
            raise Exception('API_pool: no authentications')
        for authutication in authentications:
            consumer_key = authutication["consumer_key"]
            consumer_secret = authutication["consumer_secret"]
            access_key = authutication["access_key"]
            access_secret = authutication["access_secret"]
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_key, access_secret)
            self.APIs.append(tweepy.API(auth))
            self.status.append(-1)
        self.log.d('#authentications: {}'.format(len(authentications)))

    def get_idle_api(self):
        now = time.time()
        for i,st in enumerate(self.status):
            if now >= st:
                self.log.d("use authentication {}".format(i))
                self.status[i] = -1
                return self.APIs[i]
        if self.wait:
            self.log.d("waiting 1min")
            time.sleep(60) #don't need to wait 15min because some authentication may recover earlier
            return self.get_idle_api()
        else:
            raise API_pool.AllBusyError()
            return None

    def set_busy(self, api):
        i = self.APIs.index(api)
        self.status[i] = time.time() + 15*60 #15 mins

    def parse_tweet(self, tweet):
        return {'tweet_id': tweet.id, 'created_at': tweet.created_at, 'text': tweet.text, 'hashtags': tweet.entities.get('hashtags')}

    def parse_tweets(self, tweets, user_id):
        tweets = [self.parse_tweet(t) for t in tweets]
        for tweet in tweets:
            tweet['user_id'] = user_id
        return tweets

    def fetch_tweets(self, user_id, latest):
        api = None
        try:
            if latest is None :
                api = self.get_idle_api()
                new_tweets = api.user_timeline(user_id = user_id, count = 200)
            else:
                new_tweets = []
                finished = False
                max_id = None
                while not finished:
                    api = self.get_idle_api()
                    if max_id is not None:
                        page = api.user_timeline(user_id = user_id, count = 200, max_id = max_id)
                    else:
                        page = api.user_timeline(user_id = user_id, count = 200, since_id = latest-1) #return less than or equal to latest
                    page.sort(key = lambda x:x.id, reverse=True) #smaller index has larger id
                    if len(page) == 0:
                        break
                    self.log.d("get {}'s tweets page {} {}-{} latest: {}".format(user_id, len(page), page[0].id if page else None, page[-1].id if page else None, latest))
                    if max_id is not None and page[-1].id >= max_id:
                        self.log.d('fetch tweet error old_max_id={} <= new_max_id={}'.format(max_id, page[-1].id))
                        break
                    max_id = page[-1].id
                    for t in page:
                        if t.id > latest:
                            new_tweets.append(t)
                        else:
                            finished = True
                            break
                    if len(page) < 200:
                        break
            new_tweets = self.parse_tweets(new_tweets, user_id)
            self.log.d("get {}'s tweets {} {}-{} latest: {}".format(user_id, len(new_tweets), new_tweets[0]['tweet_id'] if new_tweets else None, new_tweets[-1]['tweet_id'] if new_tweets else None, latest))
            self.log.d("get {}'s tweets {} {}-{} latest: {}".format(user_id, len(new_tweets), new_tweets[0]['created_at'] if new_tweets else None, new_tweets[-1]['created_at'] if new_tweets else None, latest))
            return new_tweets
        except tweepy.RateLimitError:
            self.log.d('RateLimitError')
            self.set_busy(api)
            return self.fetch_tweets(user_id, latest)

def build_targets(democrats, republicans):
    random.shuffle(democrats) #in case all keys get RateLimitError
    random.shuffle(republicans)
    targets = []
    i = 0
    while True:
        if i >= len(democrats) and i >= len(republicans):
            break
        if i < len(democrats):
            targets.append({'user_id': democrats[i],   'affiliation': 'democrats' })
        if i < len(id_list.republicans):
            targets.append({'user_id': republicans[i], 'affiliation': 'republicans' })
        i += 1
    return targets

def get_section(l, i, j):
    print(i,j)
    assert i >= 0 and j >= 0 and i < j
    part_size = float(len(l))/j
    return l[int(part_size*i) : int(part_size*(i+1))]

if __name__ == '__main__':
    log = Logger('main')
    if len(sys.argv) != 3:
        raise Exception('wrong number of arguments')
    democrats_ids = get_section(id_list.democrats, int(sys.argv[1]), int(sys.argv[2]))
    republicans_ids = get_section(id_list.republicans, int(sys.argv[1]), int(sys.argv[2]))
    log.d('democrats_ids {} {}-{}'.format(len(democrats_ids), democrats_ids[0], democrats_ids[-1]))
    log.d('republicans_ids {} {}-{}'.format(len(republicans_ids), republicans_ids[0], republicans_ids[-1]))
    targets = build_targets(democrats_ids, republicans_ids)
    log.d('#targets: {}'.format(len(targets)))
    url = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/cs4300')
    parsed_url = urlsplit(url)
    db_name = parsed_url.path[1:]
    authentications = json.loads(base64.b64decode(os.getenv('TWEET_AUTHENTICATIONS', 'W10=').encode('ascii')).decode('ascii'))
    #log.d(str(authentications))
    #import base64;base64.b64encode(json.dumps(authutications))
    #export TWEET_AUTHENTICATIONS=ABOVE_RESULT_WITHOUT_QUOT
    #authentications = [{"consumer_key": "eNfjPJT12a1aiFGaVSNnn6nTg", "consumer_secret": "wJk3RhuhUo5MFNnnLaJQIM2Q93gFeMMfWGUzoYd6z49z8Kis2w", "access_key": "717950588076601344-yDbU6iN96hMagodDyv2iqTxuiNQ7VkS", "access_secret": "OycOAMytzLXlik4qO32iLWxPoPaqNmoXlDrW6QfhhX7Vd"}]
    unigram_classifier_db = Unigram_Classifier_DB(url, db_name, ['democrats', 'republicans'])
    scrapper_meta_db = Scrapper_META_DB(url, db_name)
    classifier = Unigram_Classifier(unigram_classifier_db)
    api_pool = API_pool(authentications)
    scrapper = Scrapper(api_pool, scrapper_meta_db, lambda tweet, target: classifier.learn(tweet, target['affiliation']))
    scrapper.run(targets)
    unigram_classifier_db.flush()
    unigram_classifier_db.close()
    scrapper_meta_db.flush()
    scrapper_meta_db.close()
    log.d('main thread exits.')
