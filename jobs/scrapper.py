#!/usr/bin/python

import datetime
import tweepy
import csv
from pymongo import MongoClient
import pymongo
import sys
import math
import time
import re
import os
from collections import defaultdict
from urlparse import urlsplit
from itertools import groupby
from bitarray import bitarray
from bson.binary import Binary
import binascii
import gzip
import cPickle as pickle
import id_list
import json
import io
import base64

stop_words = set([u'i',
 u'me',
 u'my',
 u'myself',
 u'we',
 u'our',
 u'ours',
 u'ourselves',
 u'you',
 u'your',
 u'yours',
 u'yourself',
 u'yourselves',
 u'he',
 u'him',
 u'his',
 u'himself',
 u'she',
 u'her',
 u'hers',
 u'herself',
 u'it',
 u'its',
 u'itself',
 u'they',
 u'them',
 u'their',
 u'theirs',
 u'themselves',
 u'what',
 u'which',
 u'who',
 u'whom',
 u'this',
 u'that',
 u'these',
 u'those',
 u'am',
 u'is',
 u'are',
 u'was',
 u'were',
 u'be',
 u'been',
 u'being',
 u'have',
 u'has',
 u'had',
 u'having',
 u'do',
 u'does',
 u'did',
 u'doing',
 u'a',
 u'an',
 u'the',
 u'and',
 u'but',
 u'if',
 u'or',
 u'because',
 u'as',
 u'until',
 u'while',
 u'of',
 u'at',
 u'by',
 u'for',
 u'with',
 u'about',
 u'against',
 u'between',
 u'into',
 u'through',
 u'during',
 u'before',
 u'after',
 u'above',
 u'below',
 u'to',
 u'from',
 u'up',
 u'down',
 u'in',
 u'out',
 u'on',
 u'off',
 u'over',
 u'under',
 u'again',
 u'further',
 u'then',
 u'once',
 u'here',
 u'there',
 u'when',
 u'where',
 u'why',
 u'how',
 u'all',
 u'any',
 u'both',
 u'each',
 u'few',
 u'more',
 u'most',
 u'other',
 u'some',
 u'such',
 u'no',
 u'nor',
 u'not',
 u'only',
 u'own',
 u'same',
 u'so',
 u'than',
 u'too',
 u'very',
 u's',
 u't',
 u'can',
 u'will',
 u'just',
 u'don',
 u'should',
 u'now',
 u'rt'])

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

    def store_tweet(self, tweet, event, scores):
        assert len(scores) == len(self.affiliations)
        self.classifier_tweets_cache[event].append({'tweet': tweet, 'event': event, 'scores': scores})

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
                to_remove.intersection_update([x['_id'] for x in tweets[100:-100] if '_id' in x])
                for x in tweets[0:100] + tweets[-100:]:
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

    def close(self):
        self.client.close()

class Unigram_Classifier:
    def __init__(self, db):
        self.db = db

    def get_tags(self, tweet):
        text = tweet['text']
        return set({tag.strip("#") for tag in re.split("'| ", text) if tag.startswith("#")})

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
            if len(term_counts) != 2:
                print term_counts
            assert len(term_counts) == 2
            for affiliation, term_count in term_counts.items():
                other_term_count = sum([c for a,c in term_counts.items() if a != affiliation])
                other_event_count = sum([c for a,c in event_counts.items() if a != affiliation])
                other_portion = (float(other_term_count)+1)/(other_event_count+1)
                this_portion = (float(term_count)+1)/(event_counts[affiliation]+1)
                new_score = this_portion * math.log(this_portion / other_portion)
                scores[affiliation].append(new_score)
        return {k:max(v) for k,v in scores.items()}

    def learn(self, tweet, affiliation):
        terms = self.get_terms(tweet)
        events = self.get_tags(tweet)
        user_id = tweet['user_id']
        for event in events:
            self.db.update_event_inc(event, user_id, affiliation)
            for term in terms:
                self.db.update_term_inc(event, term, user_id, affiliation)
            scores = self.calculate_score(terms, event)
            if scores == None:
                continue
            self.db.store_tweet(tweet, event, scores)

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
        if row == None:
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
        self.fetch_complete = set()
        self.callback = callback

    def has_complete(self, user_id):
        return user_id in self.fetch_complete

    def set_complete(self, user_id):
        self.fetch_complete.add(user_id)

    def scrape(self, target):
        user_id = target['user_id']
        latest_tweet_id = self.scrapper_meta.get_latest_tweet_id_of_user(user_id)
        try:
            tweets = self.API_pool.fetch_tweets(user_id, latest_tweet_id)
        except tweepy.TweepError as e:
            self.log.d('<{}> TweepError: {}'.format(target['user_id'], str(e)))
            self.set_complete(user_id)
            return
        except Exception as e:
            self.log.d('<{}> Unexpected error: {}'.format(target['user_id'], sys.exc_info()[0]))
            raise e
        if len(tweets) == 0:
            self.set_complete(user_id)
            return
        for tweet in tweets:
            self.callback(tweet, target)
            self.scrapper_meta.update_latest_tweet_id_of_user(user_id, tweet['tweet_id'])

    def run(self, targets):
        i = 0
        total = len(targets)
        while total != len(self.fetch_complete):
            target = targets[i%total]
            user_id = target['user_id']
            i += 1
            if self.has_complete(user_id):
                continue
            self.log.d('run {} user_id: {}'.format(i, user_id))
            try:
                self.scrape(target)
            except API_pool.AllBusyError:
                self.log.d('AllBusyError')
                break
        self.log.d('end run')

class API_pool():
    class AllBusyError(Exception):
        pass

    def __init__(self, authentications):
        self.log = Logger("API_pool");
        self.APIs = [];
        self.status = []; #-1 is idle, time is busy
        for authutication in authentications:
            consumer_key = authutication["consumer_key"]
            consumer_secret = authutication["consumer_secret"]
            access_key = authutication["access_key"]
            access_secret = authutication["access_secret"]
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_key, access_secret)
            self.APIs.append(tweepy.API(auth))
            self.status.append(-1)

    def get_idle_api(self):
        now = time.time()
        for i,st in enumerate(self.status):
            if st == -1:
                return self.APIs[i]
            elif now >= st:
                self.status[i] = -1
                return self.APIs[i]
        self.log("use authentication {}".format(i))
        return None

    def set_busy(self, api):
        i = self.APIs.index(api)
        self.status[i] = time.time() + 15*60 #15 mins

    def parse_tweet(self, tweet):
        return {'tweet_id': tweet.id, 'created_at': tweet.created_at, 'text': tweet.text}

    def fetch_tweets(self, user_id, latest):
        api = self.get_idle_api()
        if api == None:
            raise API_pool.AllBusyError()
        try:
            if (latest == None) :
                new_tweets = api.user_timeline(user_id = user_id, count = 200)
            else:
                new_tweets = api.user_timeline(user_id = user_id, count = 200, since_id = latest)
            new_tweets = [self.parse_tweet(t) for t in new_tweets]
            for tweet in new_tweets:
                tweet['user_id'] = user_id
            self.log.d("get {}'s tweets {} {}-{}".format(user_id, len(new_tweets), new_tweets[0]['created_at'], new_tweets[-1]['created_at']))
            return new_tweets
        except tweepy.RateLimitError:
            self.log.d('RateLimitError')
            self.set_busy(api)
            return self.fetch_tweets(user_id, latest)

def build_targets(democrats, republicans):
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

if __name__ == '__main__':
    log = Logger('main')
    targets = build_targets(id_list.democrats, id_list.republicans)
    url = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/cs4300')
    parsed_url = urlsplit(url)
    db_name = parsed_url.path[1:]
    authentications = json.loads(base64.b64decode(os.getenv('TWEET_AUTHENTICATIONS', base64.b64encode('[]'))))
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
