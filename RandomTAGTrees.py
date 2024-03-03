import numpy as np
import useful
import Tree

def RandomTAGTrees(numTreesParam,numUniqueNodeParam,numChildrenParams,strlengthParam,characters,optThresh):
    while True:
        numTrees = int(np.ceil(np.random.gamma(numTreesParam[0],numTreesParam[1])))
        # Needs at least one tree to be a valid test and  does not need 10 or more trees
        # to find all potential errors.
        if numTrees > 1 and numTrees < 10:
            break

    while True:
        numUniqueTokens = int(np.ceil(np.random.gamma(numUniqueNodeParam[0],numUniqueNodeParam[1])))
        if numUniqueTokens > numTrees:
            break
    
    while True:
        randN = np.random.rand(numUniqueTokens)
        type = np.empty(numUniqueTokens,dtype='U')
        type[randN > 0.75] = 'a'
        type[np.logical_and(0.75 >= randN,randN > 0.5)] = 'n'
        type[0.50 >= randN] = 's'
        if np.sum(type == 's') > 0 and np.sum(type == 'n') > 0:
            break
    
    typeA,  = np.where(type == 'a')
    typeNA, = np.where(np.logical_or(type == 'a',type == 'n'))
    typeNS, = np.where(np.logical_or(type == 'n',type == 's'))
    
    token = [None]*numUniqueTokens
    for i in range(numUniqueTokens-1,-1,-1):
        while True:
            match = False
            token[i] = ''.join([characters[int(np.floor(len(characters)*np.random.rand()))] \
                                            for j in range(0,int(np.ceil(np.random.gamma(strlengthParam[0],strlengthParam[1]))))])
            for j in range(numUniqueTokens-1,i,-1):
                if token[i] == token[j]:
                    match = True
                    break
            if (not match):
                break
    
    cT = [None]*numTrees
    for i in range(0,numTrees):
        while True:
            stack = []
            leaf = []
            
            k = typeNA[int(np.floor(len(typeNA)*np.random.rand()))]
            
            rootToken = token[k]; rootType = type[k]
            cT[i] = Tree.Tree({'token':rootToken,'type':rootType})
            iter = cT[i].GetIter()
            
            numChild = int(np.ceil(np.random.gamma(numChildrenParams[0],numChildrenParams[1])))
            iter.SetChildElems([None]*numChild)
            stack.extend([iter.GetChildIter(j) for j in range(0,numChild)])
            
            while bool(stack):
                iter = stack.pop()
                numChild = int(np.floor(np.random.gamma(numChildrenParams[0],numChildrenParams[1])))
                
                if numChild == 0 or len(typeA) == 0: # Leaf
                    k = typeNS[int(np.floor(len(typeNS)*np.random.rand()))]
                    iter.SetElem({'token':token[k],'type':type[k]})
                    leaf.append(iter)
                else: # Nonleaf
                    k = typeA[int(np.floor(len(typeA)*np.random.rand()))]
                    if np.random.rand() > optThresh:
                        typeO = 'o'
                    else:
                        typeO = 'a'
                    iter.SetElem({'token':token[k],'type':typeO})
                    iter.SetChildElems([None]*numChild)
                    stack.extend([iter.GetChildIter(j) for j in range(0,numChild)])
            
            # Makes sure that there are more than one leaves
            if len(leaf) > 1:
                break
        
        # Have a foot node
        if rootType == 'a':
            k = int(np.floor(len(leaf)*np.random.rand()))
            iter = leaf[k]
            iter.SetElem({'token':rootToken,'type':'a'})

    return {'cT':cT,'token':token,'type':type}


def ConstructRandomStatementTree(token,type,c,limitNodes):
    # pdb.set_trace()
    nTree = useful.find(type,'n')
    if len(nTree) == 0:
        return {'T':None,'nodeCount':0}
    aTree = useful.find(type,'a')
    
    T = Tree.Tree(nTree[int(np.floor(len(nTree)*np.random.rand()))])
    queue = [T.GetIter()]
    nodeCount = 1
    
    while bool(queue):
        iter = queue.pop(0)
        n = iter.GetElem()
        if bool(c[n]):
            tType = c[n]['type']
            tToken = c[n]['token']
            
            nodeCount += len(tType)
            if nodeCount > limitNodes:
                return {'T':None,'nodeCount':0}
            
            children = [None]*len(tType)
            for i in range(0,len(tType)):
                # Switch-case is better
                if tType[i] == 'n':
                    match = useful.find([token[j] for j in nTree],tToken[i])
                    if len(match) == 0:
                        return {'T':None,'nodeCount':0}
                    pick = nTree[match[int(np.floor(len(match)*np.random.rand()))]]
                    children[i] = pick
                elif tType[i] == 'a':
                    match = useful.find([token[j] for j in aTree],tToken[i])
                    if len(match) == 0:
                        return {'T':None,'nodeCount':0}
                    pick = aTree[match[int(np.floor(len(match)*np.random.rand()))]]
                    children[i] = pick
                elif tType[i] == 'o':
                    if np.random.rand() > 0.5:
                        match = useful.find([token[j] for j in aTree],tToken[i])
                        if len(match) == 0:
                            return {'T':None,'nodeCount':0}
                        pick = aTree[match[int(np.floor(len(match)*np.random.rand()))]]
                        children[i] = pick

            iter.SetChildElems(children)
            for j in range(0,len(children)):
                if not(children[j] is None):
                    queue.append(iter.GetChildIter(j))

    return {'T':T,'nodeCount':nodeCount}


def TAGFullStatement(T,cT):
    U = Tree.treefun(lambda i: TAGGetImportantParts(i,cT), T)
    stackU = Tree.TreeTraverse(U,TraverseNonleaf,[],'breadth')
    
    while bool(stackU):
        iterU = stackU.pop()
        iters = iterU.GetElem()['iters']
        for i in range(0,len(iters)):
            itercU = iterU.GetChildIter(i)
            if bool(itercU.GetElem()):
                e = iterU.GetElem()
                iter = e['iters'][i]
                
                iterp = iter.GetParentIter()
                k = [j for j in range(0,iterp.NumChild()) if iter == iterp.GetChildIter(j)]
                iterTemp = itercU.GetElem()['T'].GetIter()
                iterp.AttachAsChildReplacement(iterTemp,k[0])
                
                if iter.GetElem()['type'] != 'n': # Is an adjunction node
                    adjleaf = itercU.GetElem()['adjleaf']
                    iterp = adjleaf.GetParentIter()
                    k = [j for j in range(0,iterp.NumChild()) if adjleaf == iterp.GetChildIter(j)]
                    iterp.AttachAsChildReplacement(iter,k[0])

    return U.GetIter().GetElem()['T']


def TAGGetImportantParts(i,cT):
    if (i is None):
        return None
    else:
        output = TAGGetChildren(cT[i])
        iters = output['iters']; adjleaf = output['adjleaf']
        if (not adjleaf):
            T, iter = Tree.TreeCopy(cT[i],iters)
            return {'T':T,'iters':iter,'adjleaf':None}
        else:
            T, iters = Tree.TreeCopy(cT[i],iters+[adjleaf])
            return {'T':T,'iters':iters[:-1],'adjleaf':iters[-1]}


def TraverseNonleaf(iter,stack):
    if iter.NumChild() > 0:
        stack.append(iter)
    return stack


def TAGList2Number(e):
    if bool(e):
        return e[0]
    else:
        return None


def TAGGetChildren(T):
    data = Tree.TreeTraverse(T,GetAttachingNodes,{'l':[],'adjleaf':None},'depth')
    iters = data['l']
    adjleaf = data['adjleaf']
    
    type = [e.GetElem()['type'] for e in iters]
    token = [e.GetElem()['token'] for e in iters]
    
    walk = Tree.TreeTraverse(T,StepGetWalkIter,[],'walk')
    walk = [{'iter':e,'childIdx':TAGList2Number(useful.find(iters,e))} for e in walk]
    
    return {'iters':iters,'type':type,'token':token,'walk':walk,'adjleaf':adjleaf}


def GetAttachingNodes(iter,data):
    e = iter.GetElem()
    # Switch-case is better
    if e['type'] == 'n':
        if iter.hasParent():
            data['l'].append(iter)
    elif e['type'] == 'a':
        if iter.NumChild() == 0:
            data['adjleaf'] = iter
        elif iter.hasParent():
            data['l'].append(iter)
    elif e['type'] == 'o':
        data['l'].append(iter)
    return data


def StepGetWalkIter(iter,l,*_):
    l.append(iter)
    return l


def TAGFetchTerminal(T):
    return Tree.TreeTraverse(T,RecordTerminals,[],'depth')


def RecordTerminals(iter,l):
    e = iter.GetElem()
    if e['type'] == 's':
        l.append(e['token'])
    return l