#!/usr/bin/python

import datetime
import tweepy
import csv
from pymongo import MongoClient
import pymongo
import sys
import math
import time
import id_list
from collections import defaultdict

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
        self.client = MongoClient(url)
        self.db = self.client[db_name]
        self.classifier_tweets = self.db['unigram_classifier_tweets']
        self.classifier_meta_event = self.db['unigram_classifier_meta_event']
        self.classifier_meta_term = self.db['unigram_classifier_meta_term']
        self.affiliations = affiliations
        self.setup_index()
        self.dirty_events = set()

    def setup_index(self):
        classifier_tweets_index = [('tweet.tweet_id', pymongo.ASCENDING), ('event', pymongo.ASCENDING)]
        for affiliation in affiliations:
            classifier_tweets_index.append(('scores.'+affiliation, pymongo.ASCENDING))
        self.classifier_tweets.create_index(classifier_tweets_index)
        self.classifier_meta_event.create_index([('event', pymongo.ASCENDING), ('affiliation', pymongo.ASCENDING)])
        self.classifier_meta_term.create_index([('event', pymongo.ASCENDING), ('term', pymongo.ASCENDING), ('affiliation', pymongo.ASCENDING)])

    def update_event_inc(self, event, user_id, affiliation):
        self.classifier_meta_event.update_one({'event':event, 'affiliation':affiliation}, {"$addToSet": {'user_ids': [user_id]}}, upsert=True)

    def update_term_inc(self, event, term, user_id, affiliation):
        self.classifier_meta_term.update_one({'event':event, 'term':term, 'affiliation':affiliation}, {"$addToSet": {'user_ids': [user_id]}}, upsert=True)

    def get_event_meta(self, event):
        pipline = [{'$match'  : {'event': event}},
                   {'$project': {'affiliation': 1, 'count': { '$size': '$user_ids' }}}]
        return {c['affiliation']:c['count'] for c in self.classifier_meta_event.aggragate(pipline)}

    def get_term_meta(self, event, term):
        pipline = [{'$match'  : {'event': event, 'term': term}},
                   {'$project': {'affiliation': 1, 'count': { '$size': '$user_ids' }}}]
        return {c['affiliation']:c['count'] for c in self.classifier_meta_term.aggragate(pipline)}

    def store_tweet(self, tweet, event, scores):
        assert len(scores) == len(self.affiliations)
        self.dirty_events.add(event)
        self.classifier_tweets.insert({'tweet': tweet, 'event': event, 'scores': scores})

    def clear_redundancy(self):
        for event in self.dirty_events;
            tweets = list(self.classifier_tweets.find({'event': event}, ['scores']))
            to_remove = set([x['_id'] for x in tweets])
            for affiliation in self.affiliations:
                tweets.sort(key = lambda x: x['scores'][affiliation])
                to_remove.intersection_update([x['_id'] for x in tweets[100:-100]])
            self.classifier_tweets.remove({'_id':{'$in':list(to_remove)}})

    def close(self):
        self.client.close()

class Unigram_Classifier:
    def __init__(self, db):
        self.db = db

    def get_tags(self, tweet):
        text = tweet['text']
        return set({tag.strip("#") for tag in text.split() if tag.startswith("#")})

    def clean_text(self, text):
        return re.sub(r"http\S+", "", text)

    def get_terms(self, tweet):
        text = clean_text(weet['text'])
        terms = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", text)
        terms = [t.lower() for t in terms]
        terms = [t for t in terms if t not in stop_words]
        return set(terms)

    def calculate_score(self, terms, event):
        scores = defaultdict(float)
        event_counts = self.db.get_event_meta(event)
        if any([c < 250 for c in event_counts.values()])
            return None
        for t in terms:
            term_counts = self.db.get_term_meta(event, t)
            for affiliation, term_count in term_counts.items():
                other_term_count = sum([c for a,c in term_counts.items() if c != affiliation])
                other_event_count = sum([c for a,c in event_counts.items() if c != affiliation])
                other_portion = float(other_term_count)/other_event_count
                this_portion = float(term_count)/event_counts[affiliation]
                new_score = this_portion * math.log(this_portion / other_portion)
                scores[affiliation] = max(scores[affiliation], new_score)
        return scores

    def learn(self, tweet, affiliation):
        terms = self.get_terms(tweet)
        events = self.get_tags(tweet)
        user_id = tweet['user_id']
        for e in events:
            self.db.update_event_inc(event, user_id, affiliation)
            for t in terms:
                self.db.update_term_inc(event, term, user_id, affiliation)
            scores = calculate_score(terms, e)
            if scores == None:
                continue
            self.db.store_tweet(tweet, e, scores)

class Scrapper_META_DB:
    def __init__(self, url, db_name):
        self.client = MongoClient(url)
        self.db = self.client[db_name]
        self.scrape_meta_collection = self.db['scraper_meta']

    def store_latest_tweet_id_of_user(user_id, tweet_id):
        self.scrape_meta_collection.update_one({'user_id':user_id}, {"$set": {'latest_tweet_id': tweet_id}}, upsert=True)

    def get_latest_tweet_id_of_user(user_id):
        return self.scrape_meta_collection.find_one({'user_id':user_id})['latest_tweet_id']

    def close():
        self.client.close()

class Scrapper:
    def __init__(self, API_pool, scrapper_meta, callback):
        self.log = Logger(name)
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
            self.log.d('<{}> TweepError: {}'.format(target['id'], str(e)))
            self.set_complete(user_id)
            return
        except:
            self.log.d('<{}> Unexpected error: {}'.format(target['id'], sys.exc_info()[0]))
            raise
        if len(tweets) == 0:
            self.set_complete(user_id)
            return
        for tweet in tweets:
            self.callback(tweet, target)

    def run(self, targets):
        i = 0
        total = len(targets)
        while len(targets) != len(fetch_complete):
            target = target_ids[i%total]
            user_id = target['user_id']
            i += 1
            if self.has_complete(user_id):
                continue
            self.log.d('run user_id: {}'.format(user_id))
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
        for i,st in enumerate(self.APIs):
            if st == -1:
                return self.APIs[i]
            elif now >= st:
                self.status[i] = -1
                return self.APIs[i]
        return None

    def set_busy(self, api):
        i = self.APIs.index(api)
        self.status[i] = time.time() + 15*60 #15 mins

    def parse_tweet(self, tweet):
        return {'tweet_id': tweet.id, 'created_at': tweet.created_at, 'text': tweet.text}

    def fetch_tweets(self, user_id, latest):
        api = self.get_idle_api()
        if api == None:
            raise AllBusyError()
        try:
            if (latest == None) :
                new_tweets = self.api.user_timeline(user_id = user_id, count = 200)
            else:
                new_tweets = self.api.user_timeline(user_id = user_id, count = 200, since_id = latest)
            new_tweets = [self.parse_tweet(t) for t in new_tweets]
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
            targets.append({'user_id': democrats[i],   'affiliation': 'democrats' }))
        if i < len(id_list.republicans):
            targets.append({'user_id': republicans[i], 'affiliation': 'republicans' }))
        i += 1
    return targets

if __name__ == '__main__':
    log = Logger('main')
    targets = build_targets(id_list.democrats, id_list.republicans)
    url = None
    db_name = None
    authentications = None
    unigram_classifier_db = Unigram_Classifier_DB(url, db_name, ['democrats', 'republicans'])
    scrapper_meta_db = Scrapper_META_DB(url, db_name)
    classifier = Unigram_Classifier(unigram_classifier_db)
    api_pool = API_pool(authentications)
    scrapper = Scrapper(api_pool, scrapper_meta_db, lambda tweet, target: classifier.learn(tweet, target['affiliation']))
    scrapper.run(targets)
    unigram_classifier_db.clear_redundancy()
    unigram_classifier_db.close()
    scrapper_meta_db.close()
    log.d('main thread exits.')
