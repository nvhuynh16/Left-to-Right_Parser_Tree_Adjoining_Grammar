import time
import numpy as np

import useful
from randweb import randweb

import RandomTAGTrees
import TAGParser
import TAGHelper

seed = randweb(lowerbnd = 5000000, upperbnd = 10000000)[0]
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
    
    print("Construction of random TAG tree complete.")
        
    # %% ----------------------------------------------------------------------
    O = RandomTAGTrees.TAGFullStatement(t,cT)
    terminal = RandomTAGTrees.TAGFetchTerminal(O)
    distinguished = [O.GetElem()["token"]]
    
    ##  ##
    tag = TAGParser.TAGParser(cT,distinguished)
    
    t1 = time.time()
    
    ##  ##
    tooMany = False
    for i in range(0,len(terminal)):
        print("At terminal " + str(i+1))
        tag.Input(terminal[i])
        tooMany = (tag.NumState() > 500)
        if tooMany:
            print("Too many possibilities. Skipping.\n")
            break
    
    if (tooMany):
        continue
    
    # %% ----------------------------------------------------------------------
    print("Performing comparison with " + str(tag.NumState()) + " tree(s).")
    if tag.NumState() == 0:
        cTL = [TAGHelper.Tree2Linear(e) for e in cT]
        tL = TAGHelper.Tree2Linear(t)
        
        import pickle
        with open('err'+str(seed)+'.p','wb') as f:
            pickle.dump({'cT':cT,'t':t,'cTL':cTL,'tL':tL,'terminal':terminal,
                         'distinguished':distinguished}, f)
        raise Exception("Failed to find completion.")
        break
    else:
        sT = tag.Terminate()
        b = useful.find(sT,t)
        if len(b) == 0:
            cTL = [TAGHelper.Tree2Linear(e) for e in cT]
            tL = TAGHelper.Tree2Linear(t)
            
            import pickle
            with open('err'+str(np.mod(seed,2**32))+'.p','wb') as f:
                pickle.dump({'cT':cT,'t':t,'cTL':cTL,'tL':tL,'terminal':terminal,
                             'distinguished':distinguished}, f)
            raise Exception("There is no parsed statement tree that matches the actual statement tree.")
        elif len(b) > 1:
            raise Exception("There are multiple parsed statement trees that match the actual statement.")
        else:
            count += 1
            print("Finished " + str(count) + " at " + str((time.time()-t0)/60) + " mins.")
            print("Took " + str(time.time()-t1) + " secs.\n")