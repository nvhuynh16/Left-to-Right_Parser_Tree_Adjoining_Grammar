import time
import sys

import numpy as np

import randomdotorg

import RandomTAGTrees
import TAGParser

randProfile = randomdotorg.RandomDotOrg("tag")
seed = randProfile.get_seed()
np.random.seed(int(np.mod(seed,2**32)))

t0 = time.time()
numTries = 16

count = 0
while time.time()-t0 < 24*60*60:
    t = None; nodeCount = 0

    while True:
        # Construct a random grammar
        output = RandomTAGTrees.RandomTAGTrees([2,3],[2,3],[2,0.6],[2,0.9],'abc',0.4)
        cT = output['cT'] # Get the TAG
        rootToken = [e.GetIter().GetElem()['token'] for e in cT]
        rootType = [e.GetIter().GetElem()['type'] for e in cT]
        
        cTinfo = [RandomTAGTrees.TAGGetChildren(e) for e in cT]
        
        for i in range(10): # Try at most 10 times to construct a derived tree
            attempt = RandomTAGTrees.ConstructRandomStatementTree(rootToken,rootType,cTinfo,10)
            
            if attempt['T'] and attempt['nodeCount'] < 5:
                t = attempt['T']
                nodeCount = attempt['nodeCount']
                break
        if t: # If a valid grammar and statement tree was correctly constructed,
              # then we are ready to test. Otherwise, restart from the while loop.
            break
        
    # %% ----------------------------------------------------------------------
    O = RandomTAGTrees.TAGFullStatement(t,cT)
    terminal = RandomTAGTrees.TAGFetchTerminal(O)
    distinguished = [O.GetElem()["token"]]
    
    ##  ##
    tag = TAGParser.TAGParser(cT,distinguished)
    
    # -------------
    
    sys.stdout.write("Sample number %d    \r\n" % (nodeCount) )
    sys.stdout.flush()