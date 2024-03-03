from urllib import request
import re

def randweb(n = 1, lowerbnd = 1, upperbnd = 100):
# Inspired by Ameya Deoras' "True Random Integer Generator" from MATLAB
# file exchange. The numbers are fetched from random.org.
    if lowerbnd < -1e9 or upperbnd > 1e9:
        raise Exception("The range of values must lie in [-1e9, 1e9]")

    url = 'https://www.random.org/integers/?' \
            + 'num=' + str(n) \
            + '&min=' + str(lowerbnd) \
            + '&max=' + str(upperbnd) \
            + '&col=1&base=10&format=plain&rnd=new'
    f = request.urlopen(url)
    stream = f.read()
    
    l = re.findall("-?\d+",str(stream))
    
    return [int(s) for s in l]