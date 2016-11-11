""" Python script to run multiple test for the correlation study
    Author: Zander Blasingame
    Organization: CAMEL at Clarkson University """

import subprocess
import numpy as np
import json


# Helper function
def parse_stats(input_data):
    lines = input_data.split('\n')

    return json.loads(''.join(lines[lines.index('JSON out') + 1:]))

# params
num_tests = 100

data = []
out_data = {}

for i in range(num_tests):
    proc = subprocess.Popen(['python', 'main.py',
                             '--train_file', 'data.csv',
                             '--test_file', 'data.csv'],
                            stdout=subprocess.PIPE)

    data.append(parse_stats(proc.stdout.read().decode('utf-8')))

# Get average ranking
hpcs = [item['name'] for item in data[0]]

out_data['avg'] = [dict(name=hpc, rank=sum([item['rank']
                                            for entries in data
                                            for item in entries
                                            if item['name'] == hpc])/len(data))
                   for hpc in hpcs]

out_data['entries'] = data

with open('data.json', 'w') as f:
    json.dump(out_data, f, sort_keys=True, indent=4)
