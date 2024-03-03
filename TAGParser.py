import useful

import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix, identity
import Tree

class TAGParser:
    def __init__(s,T,distinguished):
        # "T" is a list of tree-adjoining grammar trees
        # and "distinguished" is a list of strings that
        # are of the distinguished tokens.
        s.PreprocessTAGTrees(T,distinguished)
        s.Next, s.End, s.Tmatch = None, None, None
        
    def RenewParse(s):
        # Restarts the parsing process.
        s.Next, s.End, s.Tmatch = None, None, None
        
    def NumState(s):
        # Returns the number of statement trees being constructed
        # from the tokens that have been fed.
        return len(s.Tmatch)
        
    def Predict(s):
        # Returns a list of tokens that can be inputted as a follow-up
        # to the already inputted tokens.
        if s.Tmatch is None: # No strings have been parsed
           return [s.token[i] for i in s.start]
        else: # Find possible next constructions
            if s.Next is None:
                s.Next = []
                s.End = []
                for i in range(0,len(s.Tmatch)):
                    L, _ = s.DirectionConstruction(s.Tmatch[i]['iter'])
                    s.Next.extend([{'op':e,'iter':s.Tmatch[i]['iter'],
                                    'T':s.Tmatch[i]['T']} for e in L])
                    s.End.extend([{'op':e,'iter':s.Tmatch[i]['iter'],
                                    'T':s.Tmatch[i]['T']} for e in L])
            return useful.unique([s.token[e['op'][-1]] for e in s.Next])
            
    def Input(s,token):
        # Takes a token into the parser. A function call to the s.Predict()
        # function is recommended to obtain all the possible tokens that can
        # be fed into the s.Input() function. The s.Predict() also stores the
        # follow-up construction directions for future use and s.Input()
        # requires these construction directions. Hence, if no follow-up
        # construction directions are stored, s.Input() will essentially
        # perform s.Predict() anyways.
        if s.Tmatch is None: # No strings have been parsed
            s.Tmatch = []
            for i in s.start:
                if s.token[i] == token:
                    Tstart = Tree.Tree(i)
                    s.Tmatch.append({'T':Tstart,'iter':Tstart.GetIter()})
        else: # Find possible next construction directions
            if s.Next is None:
                s.Next = []
                for i in range(0,len(s.Tmatch)):
                    L, C = s.DirectionConstruction(s.Tmatch[i]['iter'])
                    s.Next.extend([{'op':e,'iter':s.Tmatch[i]['iter'],
                                    'T':s.Tmatch[i]['T']} for e in L])
            s.Tmatch = []
            for j,e in enumerate(s.Next):
                if token == s.token[e['op'][-1]]:
                    T, iter = TAGConstructForward(e['iter'], e['T'],
                                                  e['op'], s.childNum)
                    s.Tmatch.append({'T': T,'iter':iter})
        
        s.Next, s.End = None, None
        return bool(s.Tmatch)
        
    def StatementTrees(s):
        # Gets the current statement trees.
        return [Tree.ITree(e['T']) for e in s.Tmatch]
        
    def Terminate(s):
        # Gets the terminating statement tree, if any.
        if s.End is None:
            s.End = []
            for i in range(0,len(s.Tmatch)):
                _, C = s.DirectionConstruction(s.Tmatch[i]['iter'])
                s.End.extend([{'op':e,'iter':s.Tmatch[i]['iter'],
                               'T':s.Tmatch[i]['T']} for e in C])
        
        term_tree = [None]*len(s.End)
        for i in range(0,len(s.End)):
            T, _ = TAGConstructForward(s.End[i]['iter'],s.End[i]['T'],
                                       s.End[i]['op'],s.childNum)
            term_tree[i] = Tree.treefun(lambda x: None if x is None
                                             else useful.find(s.NN,x+1)[0], T)
        
        return term_tree

    
    def PreprocessTAGTrees(self,T,distinguished):
        # Convert the set of TAG elementary trees to a set of sparse matrices,
        # arrays and numbers used as preprocessing for TAG parsing.
        # 1-norm.
        denom = lambda M: max(1,2*M.sum(0).max())
        
        ##
        # Decomposes the TAG tree for efficient parsing of strings
        comp = [TAGDecomp(x) for x in T]
        
        ## Find \bar{E_A} trees
        rootToken = np.array([e['rootIter'].GetElem()['token'] for e in comp])
        rootType = np.array([e['rootIter'].GetElem()['type'] for e in comp])
        
        isA = np.where(rootType == 'a')[0]
        isE_A = [e['fpath'][-1].GetElem()['type'] == 'a' for e in comp]
        idxE_A = np.where(isE_A)[0]
        notshortE_A = [(not isE_A[i] or len(c['fpath']) > 1)
                                                for i,c in enumerate(comp)]
        
        lenA = len(idxE_A)
        A = np.zeros((lenA,lenA))
        a0 = np.zeros(lenA)
        count = np.empty(lenA, dtype = 'i')
        for j,idx in enumerate(idxE_A):
            if len(comp[idx]['adj0']) == 1:
                a0[j] = 1
            else:
                adj0 = comp[idx]['adj0'][:-1]
                count[j] = len(np.unique([e.GetElem()['token'] for e in adj0]))
                for a in adj0:
                    match = [i for i,x in enumerate(idxE_A)
                               if a.GetElem()['token']
                                    == rootToken[x] ]
                    A[(match,j)] = 1/count[j]
        a = condPath(a0,A,count)
        
        isbarE_A = np.array(isE_A)
        isbarE_A[idxE_A[~a]] = False        
        
        ## Construct the labels
        nn = np.array([1+len(e['flat']) for e in comp])
        NN = np.cumsum(nn)
        zeroNN = np.concatenate(([0],NN))
        Label = [[None]*e for e in nn]
        for i,(c,L) in enumerate(zip(comp,Label)):
            L[-1] = {'node':c['rootIter'],'side':'R',
                            'childIdx':None,'pIdx':zeroNN[i]}
            childCount = 0
            stack = []
            k = 0
            for j,f in enumerate(c['flat']):
                if isinstance(f,Tree.TreeIter):
                    ## A Switch-Case is better
                    Type = f.GetElem()['type']
                    if Type == 's':
                        L[j+k] = {'node':f,'side':'M',
                                         'childIdx':None,'pIdx':None}
                    elif Type == 'n':
                        if f.NumChild() == 0: # Is substitution
                            L[j+k] = {'node':f,'side':'M',
                                             'childIdx':childCount,'pIdx':None}
                            childCount = childCount + 1
                        else:
                            L[j+k] = {'node':f,'side':'L',
                                             'childIdx':childCount,'pIdx':None}
                            k = k + 1
                            L[j+k] = {'node':f,'side':'R',
                                             'childIdx':childCount,'pIdx':None}
                            childCount = childCount + 1
                    elif Type == 'a':
                        if f.NumChild() == 0: # Is a foot node
                            L[j+k] = {'node':f,'side':'M',
                                             'childIdx':None,'pIdx':None}
                        else: # Is adjunction node
                            L[j+k] = {'node':f,'side':'L',
                                             'childIdx':childCount,'pIdx':None}
                            stack.append({'childCount':childCount,'pIdx':j+k})
                            childCount = childCount + 1
                    elif Type == 'o':
                        L[j+k] = {'node':f,'side':'L',
                                         'childIdx':childCount,'pIdx':None}
                        stack.append({'childCount':childCount,'pIdx':j+k})
                        childCount = childCount + 1
                else:
                    e = stack.pop()
                    L[j+k] = {'node':c['flat'][f],
                                     'side':'R','childIdx':e['childCount'],
                                     'pIdx':zeroNN[i]+e['pIdx']}
                    L[e['pIdx']]['pIdx'] = zeroNN[i]+j+k
        
        label = [e for l in Label for e in l]
        token = np.array([e['node'].GetElem()['token'] for e in label])
        Type = np.array([e['node'].GetElem()['type'] for e in label])
        typeS = (Type == 's')
        side = np.array([e['side'] for e in label])
        isleafAdj = np.array([bool(e['leafAdj']) for e in comp])
        
        numlabel = len(label)
        
        childNum = np.zeros(numlabel, dtype = 'i')
        for i in range(len(NN)):
            childNum[zeroNN[i]:NN[i]-1] \
                = max([useful.None2neg1(e['childIdx'])+1 for e in Label[i]])
        
        # \gamma in the paper
        nStart = zeroNN[:-1][(rootType == 'n') & (rootToken == distinguished)]
        
        foot = np.where((side == 'M') & (Type == 'a'))[0]
        isEnd = csc_matrix((np.ones_like(NN), (NN-1, np.zeros_like(NN))),
                            shape=(numlabel,1))
        
        s0 = np.empty(numlabel)
        for i in range(len(Label)):
            s0[zeroNN[i]:zeroNN[i+1]] = zeroNN[i]
        s = [e['pIdx'] for e in label]
        
        f = np.full(numlabel,np.NaN)
        for i,(t,L) in enumerate(zip(rootType,Label)):
            if t == 'a':
                j = np.where((Type[zeroNN[i]:zeroNN[i]+len(L)] == 'a')
                             & [l['side'] == 'M' for l in L]
                            )[0] + zeroNN[i]
                f[zeroNN[i]:zeroNN[i+1]] = j
                
        barf = np.full(numlabel, -1, dtype = 'i')
        for i,(b,L) in enumerate(zip(isE_A,Label)):
            if b:
                j = np.where((Type[zeroNN[i]:zeroNN[i]+len(L)] == 'a')
                                 & [l['side'] == 'M' for l in L]
                             )[0] + zeroNN[i]
                barf[zeroNN[i]:zeroNN[i+1]] = j
        
        eps = [(-1 if e['childIdx'] is None else e['childIdx']) for e in label]
        
        
        ## Construct I, E, and N
        Eye = identity(numlabel, format = 'csc')
        
        Irow, Icol = [], []
        Erow, Ecol = [], []
        for i,L in enumerate(Label):
            for j in range(len(L)-1):
                ## A Switch-Case is better
                sideSample = L[j]['side']
                if sideSample == 'L':
                    typeSample = Type[zeroNN[i]+j]
                    if typeSample == 'a':
                        if np.any(isbarE_A & (Type[NN-1] == 'a')
                                   & (token[NN-1] == token[zeroNN[i]+j])):
                            Irow.append(zeroNN[i]+j)
                            Icol.append(zeroNN[i]+j+1)
                        else:
                            Irow.append(zeroNN[i]+j)
                            Icol.append(zeroNN[i]+j)
                            
                        
                        k = np.where(notshortE_A & (Type[NN-1] == 'a')
                                     & (token[NN-1] == token[zeroNN[i]+j]))[0]
                        
                        Erow.extend(np.full(len(k),zeroNN[i]+j))
                        Ecol.extend(zeroNN[k])
                        
                    elif typeSample == 'o':
                        Irow.append(zeroNN[i]+j)
                        Icol.append(zeroNN[i]+j+1)
                        
                        k = np.where(notshortE_A & (Type[NN-1] == 'a')
                                     & (token[NN-1] == token[zeroNN[i]+j]))[0]
                        
                        Erow.extend(np.full(len(k),zeroNN[i]+j))
                        Ecol.extend(zeroNN[k])
                                
                elif sideSample == 'M':
                    Irow.append(zeroNN[i]+j); Icol.append(zeroNN[i]+j)
                    if Type[zeroNN[i]+j] == 'n':
                        k = np.where(notshortE_A & (Type[NN-1] == 'n')
                                     & (token[NN-1] == token[zeroNN[i]+j]))[0]
                        
                        Erow.extend(np.full(len(k),zeroNN[i]+j))
                        Ecol.extend(zeroNN[k])
                        
                elif sideSample == 'R':
                    typeSample = Type[zeroNN[i]+j]
                    if typeSample == 'a':
                        Irow.append(zeroNN[i]+j)
                        Icol.append(zeroNN[i]+j)
                        
                        k = np.where(isbarE_A[isleafAdj]
                                      & (token[foot] == token[zeroNN[i]+j]))[0]
                        
                        Erow.extend(np.full(len(k),zeroNN[i]+j))
                        Ecol.extend(foot[k]+1)
                    elif typeSample == 'o':
                        Irow.append(zeroNN[i]+j)
                        Icol.append(zeroNN[i]+j+1)

                        k = np.where(isbarE_A[isleafAdj]
                                      & (token[foot] == token[zeroNN[i]+j]))[0]
                        
                        Erow.extend(np.full(len(k),zeroNN[i]+j))
                        Ecol.extend(foot[k]+1)
                        
        I = csc_matrix((np.ones_like(Irow), (Irow,Icol)),
                       shape=(numlabel,numlabel))
        E = csc_matrix((np.ones_like(Erow), (Erow,Ecol)),
                       shape=(numlabel,numlabel))
        # maximum would be better but currently no efficient version for sparse
        N = I + E
        barI = inv(Eye-I/denom(I))
        barN = inv(Eye-N/denom(N))

        
        ## P_1
        row, col = [], []
        for z, L in zip(zeroNN, Label):
            row.extend(z + np.arange(len(L)-1))
            col.extend(z + np.arange(1,len(L)))
            
        P1 = csc_matrix((np.ones_like(row), (row,col)),
                        shape=(numlabel,numlabel))
        
        
        ## R_A
        row, col = [], []
        for i in np.where(isE_A)[0]:
            for j in range(len(comp[i]['fpath']), len(Label[i])):
                k = zeroNN[i]+j
                
                if (label[k]['side'] == 'R'):
                    match = (rootToken == token[k]) & (rootType == 'a')
                    
                    pIdx = label[k]['pIdx']
                    
                    row.extend(NN[match]-1)
                    col.extend(np.full(len(row)-len(col), pIdx, dtype = 'i'))
                else:
                    break
                if (Type[k] == 'a'):
                    break
        RA = csc_matrix((np.ones_like(row), (row,col)),
                        shape=(numlabel,numlabel))
        
        
        ## F_A
        row, col = [], []
        for i in foot:
            for start,c in zip(zeroNN, comp):
                for k in range(start, start+len(c['fpath'])-1):
                    if barI[(start,k)] > 0:
                        if token[i] == token[k]:
                            row.append(i)
                            col.append(k)
                    else:
                        break
        FA = csc_matrix((np.ones_like(row), (row,col)),
                        shape=(numlabel,numlabel))
        
        
        ## F_B
        row, col = [], []
        for i in foot:
            for j in np.where(isbarE_A)[0]:
                start = zeroNN[j]+len(comp[j]['fpath'])
                k = start
                while True:
                    if barI[start,k] > 0:
                        if token[i] == token[k] and side[k] == 'L':
                            row.append(i)
                            col.append(k)
                        k = k + 1
                    else:
                        break
        FB = csc_matrix((np.ones_like(row), (row,col)),
                        shape=(numlabel,numlabel))
        
        
        ## A_S
        row, col = [], []
        for i in np.where(side == 'L')[0]:
            col.extend(zeroNN[isA[token[NN[rootType == 'a']-1] == token[i]]])
            row.extend(np.full(len(col)-len(row), i, dtype = 'i'))
        AS = csc_matrix((np.ones_like(row), (row,col)),
                        shape=(numlabel,numlabel))
        
        
        ## A_f
        index = np.where(~np.isnan(f))[0]
        Af = csc_matrix((np.ones_like(index), (index,index)),
                        shape=(numlabel,numlabel))
        
        
        ## The rest of the matrices
        FN = barI*FA*P1
        barFN = inv(Eye-FN/denom(FN))
        EbarN = E*barN
        
        gammabarN = barN[nStart,:].sum(0)
        gammabarNP1 = gammabarN*P1
        gammabarN = csc_matrix(gammabarN)
        gammabarNP1 = csc_matrix(gammabarNP1)
        
        EbarNP1 = E*barN*P1
        barFNAfFB = barFN*Af*FB
        
        start = [i for i in range(0,numlabel) if typeS[i] and gammabarN[(0,i)] > 0]
        
        self.numLabel = numlabel; self.NN = np.array(NN); self.label = label
        self.token = np.array(token); self.type = np.array(Type); self.typeS = np.array(typeS); self.side = np.array(side)
        self.start = np.array(start); self.isEnd = isEnd; self.childNum = childNum
        
        self.eps   = np.array(eps) # -1 is when eps[i] is undefined
        self.s0    = np.array(s0)
        self.s     = np.array(s)
        self.f     = np.array(f)
        self.barf  = barf
        
        self.I       = I;    self.barI    = barI
        self.E       = E;    self.ET      = E.transpose()
        self.barN    = barN; self.EbarN   = EbarN
        self.EbarNT  = EbarN.transpose()
        self.EbarNP1 = EbarNP1
        
        self.barFN = barFN; self.barFNAfFB = barFNAfFB
        
        self.RA    = RA;    self.AS = AS;     self.Af = Af
        
        self.gbarN = gammabarN; self.gbarNP1 = gammabarNP1
    
    
    def FreeCompletion(s,op):
        # Finds the tree outline from an adjunction label or substitution
        # label to a terminal label.
        l = []
        k = op[-1]
        if s.typeS[k]:
            l.append(op)
        elif s.side[k] != 'M' or s.type[k] != 'a':
            for i in [i for i in range(0,len(s.type)) if s.EbarN[k,i] > 0 and s.typeS[i]]:
                op_copy = list(op); op_copy.extend([{'dir':'D','childIdx':s.eps[k]},i])
                l.append(op_copy)
        return l


    def DirectionConstruction(s,iter):
        # Find the next possible construction directions towards the next
        # terminal and inserted into set "L" or is inserted into "C" if
        # the statement tree can be completed. "L" and "C" are returned.
        L = []; C = []
       
        container = [{'iter':iter,'op':[iter.GetElem()+1],'pk':[]}]
        while bool(container):
            e = container.pop(0) # Fetch one of the remaining nodes
            k = e['op'][-1]
            
            # Switch-case is better
            Type = s.type[k]
            if Type == 's': # Terminal Node
                L.append(e['op'])
            elif Type == 'n':
                if s.side[k] == 'M': # Substitution Node
                    for i in np.where(np.logical_and(s.type == 's',np.array((s.EbarN[k,:] > 0).todense())[0]))[0]:
                        op = list(e['op']); op.extend([{'dir':'D','childIdx':s.eps[k]},i])
                        L.append(op)
                else: # Root Node
                    iterp = e['iter'].GetParentIter()
                    if (not iterp): # Has no parent statement node
                        C.append(e['op']) # The distinguished symbol needs accounting
                        r = (s.ET[s.s0[k],:]).multiply(s.gbarN).nonzero()[1]
                        for i in r:
                           j = i + 1
                           while True:
                                op = list(e['op']); op.extend([{'dir':'U','childIdx':s.eps[i]},j])
                                if s.type[j] == 'a' and s.side[j] == 'M': # j is foot node label
                                    container.append({'iter':e['iter'],'op':op,'pk':list(e['pk'])})
                                else:
                                    L.extend(s.FreeCompletion(op))
                                if j+1 < s.numLabel and s.I[j,j+1] > 0:
                                    j += 1
                                else:
                                    break
                    else: # Has a parent statement node
                        if (not e['pk']): # If does not currently store the parent element
                            e['pk'].append(iterp.GetElem()) # store the parent element
                        m = e['pk'][-1]
                        if s.side[m] == 'R': # Side is 'R'
                            r = (s.ET[s.s0[k],:]).multiply(s.EbarN[m,:]).nonzero()[1]
                            for i in r:
                                phantom = Tree.TreeNode(i+1,iterp)
                                op = list(e['op']); op.extend([{'dir':'U','childIdx':s.eps[i]},i+1])
                                container.append({'iter':Tree.TreeIter(phantom),'op':op,'pk':list(e['pk'])})
                        else: # Side is other than 'R'
                            r = (s.ET[s.s0[k],:]).multiply(s.EbarN[m,:]).nonzero()[1]
                            for i in r:
                                j = i + 1
                                while True:
                                    op = list(e['op']); op.extend([{'dir':'U','childIdx':s.eps[i]},j])
                                    if s.type[j] == 'a' and s.side[j] == 'M': # If node is foot node
                                        container.append({'iter':e['iter'],'op':op,'pk':list(e['pk'])})
                                    else: # If node is not foot node
                                        L.extend(s.FreeCompletion(op))
                                    if j+1 < s.numLabel and s.I[j,j+1] > 0:
                                        j += 1
                                    else:
                                        break
                        if s.E[m,s.s0[k]] > 0: # No insertion, just move up
                            op = list(e['op']); op.extend([{'dir':'u','childIdx':s.eps[m]},m+1])
                            container.append({'iter':iterp,'op':op,'pk':e['pk'][:-1]})
            elif Type == 'a' or Type == 'o':
                if s.side[k] == 'L': # Is 'L' side
                    i = k
                    while (not s.typeS[i]): # while i is not a terminal
                        op = e['op'][:-1]; op.append(i)
                        if s.type[i] == 'a' and s.side[i] == 'M': # If node is foot node
                            container.append({'iter':e['iter'],'op':op,'pk':list(e['pk'])})
                        else: # If node is not foot node
                            L.extend(s.FreeCompletion(op))
                        if s.I[i,i+1] > 0:
                            i += 1
                        else:
                            break
                        if s.barI[k,i] > 0 and s.typeS[i]:
                            op = e['op'][:-1]; op.append(i)
                            L.append(op)
                elif s.side[k] == 'R': # Is 'R' side
                    if s.isEnd[k,0]: # Is root node of auxiliary tree
                        iterp = e['iter'].GetParentIter()
                        if (not e['pk']):
                            e['pk'].append(iterp.GetElem())
                        m = e['pk'][-1]
                        r = (s.RA[k,:]).multiply(s.EbarN[s.s[m],:]).nonzero()[1]
                        for i in r:
                            j = 1 + s.s[i]
                            while True:
                                op = list(e['op']); op.extend([{'dir':'U','childIdx':s.eps[i]},j])
                                L.extend(s.FreeCompletion(op))
                                if j+1 < s.numLabel and s.I[j,j+1] > 0:
                                    j += 1
                                else:
                                    break
                        if s.AS[s.s[m],s.s0[k]] > 0:
                            op = list(e['op']); op.extend([{'dir':'u','childIdx':s.eps[m]},m+1])
                            container.append({'iter':iterp,'op':op,'pk':e['pk'][:-1]})
                    else: # Is adjunction node and 'R' side
                        e['iter'].ChildrenMakeRoom(s.eps[k]+1)
                        iterc = e['iter'].GetChildIter(s.eps[k])
                        if (not iterc.GetElem()): # Current node has an empty child
                            L.extend(s.FreeCompletion(e['op']))
                            if Type == 'o': # label is of optional adjunction node
                                op = e['op'][:-1]; op.append(k+1)
                                container.append({'iter':e['iter'],'op':op,'pk':list(e['pk'])})
                        else: # Current node has a non-empty child
                            m = iterc.GetElem()
                            r = (s.EbarN[s.s[k],:]).multiply(s.EbarNT[s.s0[m],:]).nonzero()[1]
                            for i in r[s.barf[r] >= 0]:
                                j = 1 + s.barf[i]
                                while j <= s.s[i]-1:
                                    op = list(e['op']); op.extend([{'dir':'D','oldchildIdx':s.eps[k], \
                                                                    'childIdx':s.eps[i]},j])
                                    L.extend(s.FreeCompletion(op))
                                    if s.I[j,j+1] > 0:
                                        j += 1
                                    else:
                                        break
                            if s.EbarN[s.s[k],s.s0[m]] > 0:
                                pk = list(e['pk']); pk.append(k)
                                op = list(e['op']); op.extend([{'dir':'d','childIdx':s.eps[k]},m+1])
                                container.append({'iter':iterc,'op':op,'pk':pk})
                else: # Is foot node
                    if (not e['iter'].hasParent()): # Does not have a parent
                        r = (s.gbarNP1).multiply(s.barFN[k,:]).nonzero()[1]
                        for i in r:
                            j = i
                            while not (s.type[i] == 'a' and s.side[i] == 'M'): # If node is not foot node
                                op = list(e['op']); op.extend([{'dir':'U','childIdx':s.eps[i-1]},j])
                                L.extend(s.FreeCompletion(op))
                                if j+1 < s.numLabel and s.I[j,j+1] > 0:
                                    j += 1
                                else:
                                    break
                    else: # Does have a parent
                        iterp = e['iter'].GetParentIter()
                        if (not e['pk']):
                            e['pk'].append(iterp.GetElem())
                        m = e['pk'][-1]
                        r = (s.EbarNP1[m,:]).multiply(s.barFN[k,:]).nonzero()[1]
                        for i in r:
                            j = i
                            while not (s.type[i] == 'a' and s.side[i] == 'M'): # If node is not foot node
                                op = list(e['op']); op.extend([{'dir':'U','childIdx':s.eps[i-1]},j])
                                L.extend(s.FreeCompletion(op))
                                if j+1 < s.numLabel and s.I[j,j+1] > 0:
                                    j += 1
                                else:
                                    break
                        if s.side[m] == 'L':
                            if s.EbarN[m,s.s0[k]] > 0:
                                op = list(e['op']); op.extend([{'dir':'u','childIdx':s.eps[m]},m+1])
                                container.append({'iter':iterp,'op':op,'pk':e['pk'][:-1]})
                        elif s.side[m] == 'R':
                            r = (s.barFNAfFB[k,:]).multiply(s.EbarN[m,:]).nonzero()[1]
                            for j in r:
                                phantom = Tree.TreeNode(j+1,iterp)
                                op = list(e['op']); op.extend([{'dir':'U','childIdx':s.eps[j]},j+1])
                                container.append({'iter':Tree.TreeIter(phantom),'op':op,'pk':list(e['pk'])})
        return L, C


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
        if (~(x ^ y)).all():
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

