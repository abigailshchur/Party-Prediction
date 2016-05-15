# import sys
# sys.path.append('../jobs/')
# import scrapper

import os
import sys
import numpy as np
import cPickle as pickle
# job_dir = os.path.dirname(os.path.abspath(__file__)) + '/../jobs'
# sys.path.append(job_dir)
# print sys.path
# import scrapper

# print os.path.join(os.path.dirname(__file__), os.pardir, 'jobs')

if not os.path.exists(os.path.join(os.path.dirname(__file__), 'big_dickt.p')):
	big_dickt = {str(i): np.random.random(1) for i in range(200000)}
	pickle.dump(big_dickt, open('big_dickt.p', 'wb'), protocol=2)
else:
	big_dickt = pickle.load(open('big_dickt.p', 'rb'))

# pickle.dump(big_dickt, open('big_dickt.p', 'wb'))