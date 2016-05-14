# import sys
# sys.path.append('../jobs/')
# import scrapper

import os
import sys
job_dir = os.path.dirname(os.path.abspath(__file__)) + '/../jobs'
sys.path.append(job_dir)
print sys.path
import scrapper