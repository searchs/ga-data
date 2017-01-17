import os, sys
from subprocess import call

import json
import urllib3


data = []
counter = 0
with open('data.json') as f:
    for line in f:
        # data.append(json.loads(line))
        data.append(line)
        try:
            call(['curl', '-XPUT', 'localhost:9200/catalogue/product/{}?pretty'.format(counter), '-d', line])
        except Exception as e:
            print("ERROR: " + str(e))
        finally:
            counter += 1

        if counter == 5:
            break
