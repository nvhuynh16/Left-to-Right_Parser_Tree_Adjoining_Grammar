import useful
import numpy as np
import scipy.sparse.linalg as linalg
import scipy.sparse as sparse
import Tree

def TAGDecomp(T):
    # This function decomposes the TAG tree for efficient parsing of strings.
    # INPUT:
    #   T        - TAG tree.
    # OUTPUT:
    #   flat     - The flattened version of the TAG tree.
    #   fpath    - The first-child path of the TAG tree.
    #   rootIter - The iterator to the root node of the TAG tree.
    #   leafAdj  - The iterator to the leaf adjoining node. Only applies if the
    #              TAG tree is an auxillary tree. If not an auxillary tree,
    #              then leafAdj is empty.
    #   adj0     - The set of nonoptional adjoining nodes on the first-child
    #              path.

    # Setup the data structure for the Tree Walk
    data = {'flat':[],'fpath':[],'fadj':[],'prevdir':'d', \
            'first':True,'leafAdj':None,'index':-1,'adjStack':[]}
    # Perform the tree walk
    data = Tree.TreeTraverse(T,TAGTreeWalkStep,data,'walk')
    # Isolate the flattened tree, first path, and the leaf adjoining node
    rootIter = T.GetIter()
    cIdx = [e for e in data['flat'] if isinstance(e,Tree.TreeIter)]
    numChildren = len(useful.unique(cIdx)['l'])
    # Isolate the zero-th first-child path (->0) nonoptional adjunction nodes
    adj0 = [iter for iter in data['fpath'] if iter.GetElem()['type'] == 'a']
    return {'flat':data['flat'],'numChildren':numChildren,\
            'fpath':data['fpath'],'fadj':data['fadj'],'rootIter':rootIter,\
            'leafAdj':data['leafAdj'],'adj0':adj0}


def TAGTreeWalkStep(iter,data,curdir):
    if (not iter.hasParent()):
        # Skip the root node
        data['prevdir'] = 'd'
    # Adjunction node is found.
    elif iter.GetElem()['type'] == 'a' or iter.GetElem()['type'] == 'o':
        if data['prevdir'] == 'd': # If it's the first time adjunction node is encountered
            data['index'] += 1 # Update index
            data['flat'].append(iter) # Push to flat
            if data['first']:
                data['fpath'].append(iter) # Push to fpath
                if iter.GetElem()['type'] == 'a' or iter.GetElem()['type'] == 'o':
                    data['fadj'].append(iter)
            if curdir == 'u': # Leaf adjoining node is found
                data['leafAdj'] = iter
                data['first'] = False # No longer a first-child path
            else: # Otherwise, save the index of this adjunction node
                data['adjStack'].append(data['index'])
        elif curdir == 'u': # and data['prevdir'] == 'u'
            data['index'] += 1 # Update index
            # Fetch the index of adjunction node to go back to
            data['flat'].append(data['adjStack'].pop())
        data['prevdir'] = curdir
    elif data['prevdir'] == 'd' and curdir == 'u': # A leaf node is found
        data['index'] += 1 # Update index
        data['flat'].append(iter) # Push to flat
        if data['first']:
            data['fpath'].append(iter) # Push to fpath
            if iter.GetElem()['type'] == 'a' or iter.GetElem()['type'] == 'o':
                data['fadj'].append(iter)
        data['first'] = False # No longer a first-child path
        data['prevdir'] = curdir
    return data


def condPath(x0,A,denom):
    # NOTE: "> 2*denom./(2*denom+1)" is used instead of ">= 1" for its
    # robustness to round off error.
    cutoff = 2*denom/(2*denom+1)
    y = (x0 > 0)
    while True:
        x = np.maximum(np.dot(y,A) >= cutoff,y)
        if (x - y == 0).all():
            break
        else:
            y = x
    return x


def TAGConstructForward(iterT,T,op,childNum):
    # Performs the construction directions "op" on a copy of the tree
    # "T" starting with the node of TreeIter "iter".
    # Returns the tree copy.
    O, iter = Tree.TreeCopy(T,iterT)
    iter.SetElem(op[0])
    for i in range(1,len(op),2):
        next = op[i]
        e = op[i+1]
        # A Switch-case is better here
        if next['dir'] == 'u':
            iterp = iter.GetParentIter()
            iterp.SetElem(e)
            iterp.ReplaceChild(next['childIdx'],iter)
            iter = iterp
        elif next['dir'] == 'U':
            iter = iter.InsertParent(e,next['childIdx'])
            iter.ChildrenMakeRoom(childNum[e])
        elif next['dir'] == 'd':
            iter = iter.GetChildIter(next['childIdx'])
            iter.SetElem(e)
        else: # next['dir'] == 'D'
            if 'oldchildIdx' in next:
                iter = iter.InsertChild(e,next['oldchildIdx'],next['childIdx'])
            else:
                if next['childIdx'] >= iter.NumChild():
                    for j in range(iter.NumChild(),next['childIdx']):
                        iter.AddChild()
                    iter.AddChild(e)
                    iter = iter.GetChildIter(next['childIdx'])
                else:
                    iter = iter.GetChildIter(next['childIdx'])
                    iter.SetElem(e)
            iter.ChildrenMakeRoom(childNum[int(e)])
    return O, iter

def PreprocessTAGTrees(T,distinguished):
# Convert the set of TAG elementary trees to a set of sparse matrices,
# arrays and numbers used as preprocessing for TAG parsing.
    # 1-norm.
    denom = lambda M: max(1,2*M.sum(0).max())
    
    ##  ##
    # Decomposes the TAG tree for efficient parsing of strings
    c = [TAGDecomp(x) for x in T]
    
    ## Find \bar{E_A} trees
    isA = [i for i,e in enumerate(c) if e['rootIter'].GetElem()['type'] == 'a']
    isE_A = [e['fpath'][-1].GetElem()['type'] == 'a' for e in c]
    idxE_A = useful.find(isE_A)
    notshortE_A = [(not isE_A[i] or len(c[i]['fpath']) > 1) for i in range(0,len(c))]
    
    A = np.zeros((len(idxE_A),len(idxE_A)))
    a0 = np.zeros(len(idxE_A))
    count = np.zeros((len(idxE_A)))
    for j in range(0,len(idxE_A)):
        if len(c[idxE_A[j]]['adj0']) == 1:
            a0[j] = 1
        else:
            output = useful.unique([e.GetElem()['token'] for e in c[idxE_A[j]]['adj0'][:-1]])
            count[j] = len(output['l'])
            for k in range(0,len(c[idxE_A[j]]['adj0'])-1):
                match = [i for i,x in enumerate(idxE_A) \
                           if c[idxE_A[j]]['adj0'][k].GetElem()['token'] \
                                     == c[x]['rootIter'].GetElem()['token'] ]
                A[(match,j)] = 1/count[j]
    a = condPath(a0,A,count)
    
    isbarE_A = np.array(isE_A)
    idx = np.array(idxE_A)[np.logical_not(a)]
    if len(idx) > 0:
        isbarE_A[idx] = False
    idxbarE_A, = np.where(isbarE_A)
    
    
    ## Construct the labels
    nn = [1+len(e['flat']) for e in c]
    NN = list(np.cumsum(nn))
    zeroNN = [0]+NN
    Label = [[None]*e for e in nn]
    for i in range(0,len(c)):
        Label[i][-1] = {'node':c[i]['rootIter'],'side':'R','childIdx':None,'pIdx':zeroNN[i]}
        childCount = 0
        stack = []
        k = 0
        for j in range(0,len(c[i]['flat'])):
            if isinstance(c[i]['flat'][j],Tree.TreeIter):
                ## A Switch-Case is better
                Type = c[i]['flat'][j].GetElem()['type']
                if Type == 's':
                    Label[i][j+k] = {'node':c[i]['flat'][j],'side':'M','childIdx':None,'pIdx':None}
                elif Type == 'n':
                    if c[i]['flat'][j].NumChild() == 0: # Is substitution node
                        Label[i][j+k] = {'node':c[i]['flat'][j],'side':'M','childIdx':childCount,'pIdx':None}
                        childCount = childCount + 1
                    else:
                        Label[i][j+k] = {'node':c[i]['flat'][j],'side':'L','childIdx':childCount,'pIdx':None}
                        k = k + 1;
                        Label[i][j+k] = {'node':c[i]['flat'][j],'side':'R','childIdx':childCount,'pIdx':None}
                        childCount = childCount + 1
                elif Type == 'a':
                    if c[i]['flat'][j].NumChild() == 0: # Is a foot node
                        Label[i][j+k] = {'node':c[i]['flat'][j],'side':'M','childIdx':None,'pIdx':None}
                    else: # Is adjunction node
                        Label[i][j+k] = {'node':c[i]['flat'][j],'side':'L','childIdx':childCount,'pIdx':None}
                        stack.append({'childCount':childCount,'pIdx':j+k})
                        childCount = childCount + 1
                elif Type == 'o':
                    Label[i][j+k] = {'node':c[i]['flat'][j],'side':'L','childIdx':childCount,'pIdx':None}
                    stack.append({'childCount':childCount,'pIdx':j+k})
                    childCount = childCount + 1
            else:
                e = stack.pop()
                Label[i][j+k] = {'node':c[i]['flat'][c[i]['flat'][j]],
                                 'side':'R','childIdx':e['childCount'],'pIdx':zeroNN[i]+e['pIdx']}
                Label[i][e['pIdx']]['pIdx'] = zeroNN[i]+j+k
    
    label = [e for l in Label for e in l]
    token = [e['node'].GetElem()['token'] for e in label]
    Type = [e['node'].GetElem()['type'] for e in label]
    typeS = [e == 's' for e in Type]
    side = [e['side'] for e in label]
    
    childNum = [0]*len(label)
    for i in range(0,len(NN)):
        childNum[zeroNN[i]:NN[i]-1] = [max([useful.None2neg1(e['childIdx'])+1 for e in Label[i]])]*len(Label[i])
    
    # \gamma in the paper
    nStart = [zeroNN[i] for i,e in enumerate(c) \
      if e['rootIter'].GetElem()['type'] == 'n' \
         and bool(useful.find(distinguished,e['rootIter'].GetElem()['token']))]
    
    foot = [i for i in range(0,len(label)) if side[i] == 'M' and Type[i] == 'a']
    isEnd = sparse.csc_matrix(([1]*len(NN),([i-1 for i in NN],[0]*len(NN))),shape=(len(label),1))
    
    s0 = [np.NaN]*len(label)
    for i in range(0,len(Label)):
        s0[zeroNN[i]:zeroNN[i+1]] = [zeroNN[i]]*(zeroNN[i+1]-zeroNN[i])
    s = [e['pIdx'] for e in label]
    
    f = [np.NaN]*len(label)
    for i in range(0,len(c)):
        if c[i]['rootIter'].GetElem()['type'] == 'a':
            j = useful.find([Label[i][j]['side'] == 'M' and Type[zeroNN[i]+j] == 'a' \
                        for j in range(0,len(Label[i]))])[0] + zeroNN[i]
            f[zeroNN[i]:zeroNN[i+1]] = [j]*(zeroNN[i+1]-zeroNN[i])
            
    barf = np.full(len(label), -1, dtype = 'i')
    for i in range(0,len(c)):
        if isE_A[i]:
            j = useful.find([Label[i][j]['side'] == 'M' and Type[zeroNN[i]+j] == 'a' \
                        for j in range(0,len(Label[i]))])[0] + zeroNN[i]
            barf[zeroNN[i]:zeroNN[i+1]] = j
    
    eps = [(-1 if e['childIdx'] is None else e['childIdx']) for e in label]
    
    
    ## Construct I,E, and N
    Eye = sparse.eye(len(label),len(label))
    
    Irow = []; Icol = []
    Erow = []; Ecol = []
    for i in range(0,len(c)):
        for j in range(0,len(Label[i])-1):
            ## A Switch-Case is better
            sideSample = Label[i][j]['side']
            if sideSample == 'L':
                typeSample = Type[zeroNN[i]+j]
                if typeSample == 'a':
                    if any([(Type[NN[k]-1] == 'a' and isbarE_A[k] \
                          and token[NN[k]-1] == token[zeroNN[i]+j]) for k in range(0,len(c))]):
                        Irow.append(zeroNN[i]+j); Icol.append(zeroNN[i]+j+1)
                    else:
                        Irow.append(zeroNN[i]+j); Icol.append(zeroNN[i]+j)
                    for k in range(0,len(c)):
                        if Type[NN[k]-1] == 'a' and notshortE_A[k] \
                                and token[NN[k]-1] == token[zeroNN[i]+j]:
                            Erow.append(zeroNN[i]+j); Ecol.append(zeroNN[k])
                elif typeSample == 'o':
                    Irow.append(zeroNN[i]+j); Icol.append(zeroNN[i]+j+1)
                    for k in range(0,len(c)):
                        if Type[NN[k]-1] == 'a' and notshortE_A[k] \
                            and token[NN[k]-1] == token[zeroNN[i]+j]:
                            Erow.append(zeroNN[i]+j); Ecol.append(zeroNN[k])
            elif sideSample == 'M':
                Irow.append(zeroNN[i]+j); Icol.append(zeroNN[i]+j)
                if Type[zeroNN[i]+j] == 'n':
                    for k in range(0,len(c)):
                        if Type[NN[k]-1] == 'n' and token[NN[k]-1] == token[zeroNN[i]+j]:
                            Erow.append(zeroNN[i]+j); Ecol.append(zeroNN[k])
            elif sideSample == 'R':
                typeSample = Type[zeroNN[i]+j]
                if typeSample == 'a':
                    Irow.append(zeroNN[i]+j); Icol.append(zeroNN[i]+j)
                    K = np.nonzero(np.logical_and(isbarE_A[np.array([bool(e['leafAdj']) for e in c])], \
                                                        [token[k] == token[zeroNN[i]+j] for k in foot]))[0].tolist()
                    for k in K:
                        Erow.append(zeroNN[i]+j); Ecol.append(foot[k]+1)
                elif typeSample == 'o':
                    Irow.append(zeroNN[i]+j); Icol.append(zeroNN[i]+j+1)
                    K = np.nonzero(np.logical_and(isbarE_A[np.array([bool(e['leafAdj']) for e in c])], \
                                                        [token[k] == token[zeroNN[i]+j] for k in foot]))[0].tolist()
                    for k in K:
                        Erow.append(zeroNN[i]+j); Ecol.append(foot[k]+1)
    I = sparse.csc_matrix(([1]*len(Irow),(Irow,Icol)),shape=(len(label),len(label)))
    E = sparse.csc_matrix(([1]*len(Erow),(Erow,Ecol)),shape=(len(label),len(label)))
    N = I + E # maximum would be better but currently no efficient version for sparse
    barI = linalg.inv(Eye-I/denom(I))
    barN = linalg.inv(Eye-N/denom(N))

    
    ## P_1
    row = []; col = []
    for i in range(0,len(c)):
        row.extend([zeroNN[i]+j for j in range(0,len(Label[i])-1)])
        col.extend([zeroNN[i]+j for j in range(1,len(Label[i]))])
    P1 = sparse.csc_matrix(([1]*len(row),(row,col)),shape=(len(label),len(label)))
    
    
    ## R_A
    rootToken = [e['rootIter'].GetElem()['token'] for e in c]
    isaux = [e['rootIter'].GetElem()['type'] == 'a' for e in c]
    row = []; col = []
    for i in range(0,len(c)):
        if isE_A[i]:
            for j in range(len(c[i]['fpath']),len(Label[i])):
                if (label[zeroNN[i]+j]['side'] == 'R'):
                    match = [rootToken[k] == label[zeroNN[i]+j]['node'].GetElem()['token'] \
                                and isaux[k] for k in range(0,len(c))]
                    pIdx = label[zeroNN[i]+j]['pIdx']
                    for k in useful.find(match):
                        row.append(NN[k]-1)
                        col.append(pIdx)
                else:
                    break
                if (Type[zeroNN[i]+j] == 'a'):
                    break
    RA = sparse.csc_matrix(([1]*len(row),(row,col)),shape=(len(label),len(label)))
    
    
    ## F_A
    row = []; col = []
    for i in foot:
        for j in range(0,len(c)):
            start = zeroNN[j]
            for k in range(zeroNN[j],zeroNN[j]+len(c[j]['fpath'])-1):
                if barI[(start,k)] > 0:
                    if token[i] == token[k]:
                        row.append(i)
                        col.append(k)
                else:
                    break
    FA = sparse.csc_matrix(([1]*len(row),(row,col)),shape=(len(label),len(label)))
    
    
    ## F_B
    row = []; col = []
    for i in foot:
        for j in range(0,len(c)):
            if isbarE_A[j]:
                start = zeroNN[j]+len(c[j]['fpath'])
                k = start
                while True:
                    if barI[(start,k)] > 0:
                        if token[i] == token[k] and side[k] == 'L':
                            row.append(i)
                            col.append(k)
                        k = k + 1
                    else:
                        break
    FB = sparse.csc_matrix(([1]*len(row),(row,col)),shape=(len(label),len(label)))
    
    
    ## A_S
    row = []; col = []
    for i in [i for i,e in enumerate(side) if e == 'L']:
        for j in isA:
            if token[i] == token[NN[j]-1]:
                row.append(i)
                col.append(zeroNN[j])
    AS = sparse.csc_matrix(([1]*len(row),(row,col)),shape=(len(label),len(label)))
    
    
    ## A_f
    row = []; col = []
    for i in range(0,len(label)):
        if f[i] != np.NaN:
            row.append(i)
            col.append(i)
    Af = sparse.csc_matrix(([1]*len(row),(row,col)),shape=(len(label),len(label)))
    
    
    ## The rest of the matrices
    FN = barI*FA*P1
    barFN = linalg.inv(Eye-FN/denom(FN))
    EbarN = E*barN
    
    gammabarN = barN[nStart,:].sum(0)
    gammabarNP1 = gammabarN*P1
    gammabarN = sparse.csc_matrix(gammabarN)
    gammabarNP1 = sparse.csc_matrix(gammabarNP1)
    
    EbarNP1 = E*barN*P1
    barFNAfFB = barFN*Af*FB
    
    start = [i for i in range(0,len(label)) if typeS[i] and gammabarN[(0,i)] > 0]
    
#    self.numLabel = len(label); self.NN = np.array(NN); self.label = label
#    self.token = np.array(token); self.type = np.array(Type); self.typeS = np.array(typeS); self.side = np.array(side)
#    self.start = np.array(start); self.isEnd = isEnd; self.childNum = childNum
    
    return P1,I,E,N,EbarN,EbarNP1,Af,AS,RA, \
            FA,FB,FN,barFN,barFNAfFB
    
#    self.eps   = np.array(eps) # -1 is when eps[i] is undefined
#    self.s0    = np.array(s0)
#    self.s     = np.array(s)
#    self.f     = np.array(f)
#    self.barf  = barf
#    
#    self.I       = I;    self.barI    = barI
#    self.E       = E;    self.ET      = E.transpose()
#    self.barN    = barN; self.EbarN   = EbarN
#    self.EbarNT  = EbarN.transpose()
#    self.EbarNP1 = EbarNP1
#    
#    self.barFN = barFN; self.barFNAfFB = barFNAfFB
#    
#    self.RA    = RA;    self.AS = AS;     self.Af = Af
#    
#    self.gbarN = gammabarN; self.gbarNP1 = gammabarNP1

def MVRsplit(tag):
    Lt = np.where(tag.type == 's')[0] # Terminal
    Ls = np.where(np.logical_and(tag.type == 'n',tag.side == 'M'))[0] # substitution
    Ll = np.where(np.logical_and(np.logical_or(tag.type == 'a',tag.type == 'o'),
                                 tag.side == 'L'))[0] # L-side adjunction
    Lr = np.where(np.logical_and(np.logical_or(tag.type == 'a',tag.type == 'o'),
                                 tag.side == 'R'))[0] # R-side adjunction
    Li = np.where(np.logical_and(tag.type == 'n',tag.side == 'R'))[0] # initial root
    La = np.where(np.logical_and(tag.type == 'a',tag.side == 'R'))[0] # aux root
    Lf = np.where(np.logical_and(tag.type == 'a',tag.side == 'M'))[0] # foot
    
def negbool(b,A):
    return b+(-1)**b*A
    
def imply(A,B):
    np.logical_or(np.logical_not(A),B)

def MVimplication(Matrices,vectors):
    n,p,q = np.shape(Matrices)
    # (if negation, then negation, -1 0 1 row, -1 0 1 column, if matrix, then matrix)
    MatrixMatrix = np.empty((2,2,3,3,n,n),dtype=np.bool)
    
    A = np.empty(3,3,n,p-2,q-2)
    for i in range(-1,2):
        for j in range(-1,2):
            A[i,j,:,:] = Matrices[:,1+i:p-1+i,1+j:q-1+j]
    
    for i in range(2):
        for j in range(2):
            for k in range(-1,2):
                for l in range(-1,2):
                    for u in range(n):
                        for v in range(n):
                            MatrixMatrix[i,j,k,l,u,v] = \
                                np.min(imply(negbool(i,Matrices[u,1:-1,1:-1]),
                                             negbool(j,Matrices[v,1+k:-1-k,1+l:-1-l])))
    
    # -------------------------------------------------------------------------
    # (if negation, then negation, -1 0 1 shift, if vector, then vector)
    VectorVector = np.empty((2,2,3,n,n),dtype=np.bool)
    
    m,r = np.shape(vectors)
    v = np.empty(3,3,m,r-2)
    for i in range(-1,2):
        v[i,j,:] = vectors[:,1+i:p-1+i]
    
    for i in range(2):
        for j in range(2):
            for k in range(-1,2):
                for u in range(n):
                    for v in range(n):
                        VectorVector[i,j,k,l,u] = \
                            np.min(imply(negbool(i,vectors[u,1:-1])),
                                         negbool(j,vectors[v,1+k:-1-k]))