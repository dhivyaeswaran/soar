import os
from src.datasets import youtube
from src.patterns import distributions, pointvalues

if __name__ == '__main__':
    data = youtube.load_data()
    os.system('mkdir -p results')
    os.system('mkdir -p results/youtube')
    distributions.get_patterns('youtube', data)
    pointvalues.get_patterns('youtube', data)
    print 'Check results/youtube directory for results..'
