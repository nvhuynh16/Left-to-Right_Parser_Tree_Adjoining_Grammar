import numpy as np
import scipy

import useful
import Tree

def TAGFromString(S):
    # Convert string to TAG tree

    # Get the tokens from the string
    token, token_err = TAGTokenize(S)
    if not(token_err is None):
        err = {'msg':token_err['msg'], 'loc':token_err['loc']}
        return None, err
    # Check root node
    root = token[0]
    if root['type'] == 'o':
        err = {'msg':"Optional adjunction token is not used for root nodes",
               'loc':root['index']}
        return None, err
    elif not(root['type'] == 'n' or root['type'] == 'a'):
        err = {'msg':"TAG must start with a name for the root node",
               'loc':root['index']}
        return None, err
    
    # Construct the tree
    T, data, err = Tree.TreeConstruct(TAGTraverse,token)
    
    if not(token_err is None):
        err = {'msg':err['msg'], 'loc':err['loc']}
        return T, err
    # Check auxiliary tree
    if root['type'] == 'a':
        leaf = Tree.TreeTraverse(T,LeafIters,[],'depth')
        if len(leaf) < 2:
            err = {'msg':"Auxiliary TAG must have two or more leaves",
                   'loc':len(S)}
            return T, err
        match = [iter for iter in leaf if root['token'] == iter.GetElem()['token']]
        if len(match) < 1:
            err = {'msg':"Auxiliary tree does not have a foot node (a leaf with same root node name)",
                   'loc':len(S)}
        elif len(match) > 1:
            err = {'msg':"Auxiliary tree has too many foot nodes (a leaf with same root node name)",
                   'loc':len(S)}

    return Tree.ITree(T), None

def LeafIters(iter,l):
    if iter.NumChild() == 0:
        l.append(iter)
    return l

def TAGToString(T):
    return Tree.TreeTraverse(T, TAGStringAppend, ("", True), "walk")[0].replace("] [", " ")

def TAGStringAppend(iter, data, curdir):
    string, went_down = data
    
    if went_down:
        elem = iter.GetElem()
        if elem['type'] in {'a', b'a'}:
            string += '\\' + elem['token']
        elif elem['type'] in {'o', b'o'}:
            string += '\\*' + elem['token']
        elif elem['type'] in {'s', b's'}:
            string += '"' + elem['token'] + '"'
        else: # 'n'
            string += elem['token']
    
    if curdir == 'd':
        string += ' ['
    elif curdir == 'u':
        string += ']'
        
    return string, curdir == 'd'

def TAGTokenize(s):
    # Escape tokens: Adjoining: '\', Optional Adjoining:'\*', Start and end children: '[', ']'
    l = []; err = None
    quote = False; slash = False; optAdj = False
    token = ''; sbracket = 0
    for i in range(0,len(s)):
        if quote: # Inside quote
            if s[i] == '\"':
                if slash: # Replace backslash with quotation mark
                    token += '\"'
                    slash = False
                else: # Found the end quotation
                    l.append({'token':token,'type':'s','index':i})
                    quote = False
                    token = ''
            elif s[i] == '\\': # potential attempt to input quotation mark found
                if slash:
                    token += '\\'
                    slash = False
                else:
                    slash = True
            else:
                token += s[i]
                slash = False
        elif slash: # Found adjunction node
            if ('a' <= s[i] and s[i] <= 'z') or ('A' <= s[i] and s[i] <= 'Z'):
                token += s[i]
            elif ('0' <= s[i] and s[i] <= '9') or s[i] == '_':
                if (not token):
                    err = {'msg':"Grammar name must start with alphabetic character",'loc':i}
                    break
                else:
                    token += s[i]
            elif s[i] == ' ':
                if (not token):
                    print(str(i))
                    err = {'msg':"Isolated backslash is undefined in grammar",'loc':i-1}
                    break
                else:
                    if optAdj:
                        l.append({'token':token,'type':'o','index':i-len(token)})
                        optAdj = False
                    else:
                        l.append({'token':token,'type':'a','index':i-len(token)})
                    slash = False
                    token = ''
            elif s[i] == '[':
                if (not token):
                    err = {'msg':"Undefined grammar term \"[\"",'loc':i-1}
                    break
                else:
                    if optAdj:
                        l.append({'token':token,'type':'o','index':i-len(token)})
                        optAdj = False
                    else:
                        l.append({'token':token,'type':'a','index':i-len(token)})
                    token = ''
                    slash = False
                    sbracket += 1
                    l.append({'token':'[','type':'[','index':i})
            elif s[i] == ']':
                if (not token):
                    err = {'msg':"Undefined grammar term \"]\"",'loc':i-1}
                else:
                    if optAdj:
                        l.append({'token':token,'type':'o','index':i-len(token)})
                        optAdj = False
                    else:
                        l.append({'token':token,'type':'a','index':i-len(token)})
                    token = ''
                    slash = False
                    sbracket -= 1
                    if sbracket < 0:
                        err = {'msg':"Unmatched \"]\"",'loc':i}
                        break
                    l.append({'token':']','type':']','index':i})
            elif s[i] == '*' and (not token):
                optAdj = True
            else:
                err = {'msg':"Only alphanumeric term is allowed",'loc':i}
                break
        # Found Name
        elif ('a' <= s[i] and s[i] <= 'z') or ('A' <= s[i] and s[i] <= 'Z'):
            token += s[i]
        elif ('0' <= s[i] and s[i] <= '9') or s[i] == '_':
            if (not token):
                err = {'msg':"Grammar name must start with alphabetic",'loc':i}
                break
            else:
                token += s[i]
        # Found white space
        elif s[i] == ' ':
            if bool(token):
                l.append({'token':token,'type':'n','index':i-len(token)})
                token = ''
        # Found '['
        elif s[i] == '[':
            if bool(token):
                l.append({'token':token,'type':'n','index':i-len(token)})
                token = ''
            sbracket += 1
            l.append({'token':'[','type':'[','index':i})
        # Found ']'
        elif s[i] == ']':
            if bool(token):
                l.append({'token':token,'type':'n','index':i-len(token)})
                token = ''
            sbracket -= 1
            if sbracket < 0:
                err = {'msg':"Unmatched \"]\"",'loc':i}
                break
            l.append({'token':']','type':']','index':i})
        # Found start of quotation
        elif s[i] == '\"':
            quote = True
        # Found backslash
        elif s[i] == '\\':
            slash = True
        else:
            err = {'msg':"Undefined character \"" + s[i] + "\"",'loc':i}
            break
    
    # Report quotation and closure errors, if any.
    if bool(err):
        if quote:
            err = {'msg':"Quotation did not end",'loc':len(s)}
        elif sbracket > 0:
            if sbracket == 1:
                err = {'msg':"One square bracket was not closed",'loc':len(s)}
            else:
                err = {'msg':str(sbracket) + " square brackets was not closed",'loc':len(s)}
    return l, err


def TAGTraverse(l):
    # Ouput:
    #       elem -
    #       move -
    #       data -
    if len(l) > 0:
        elem = l.pop(0)
    else:
        return 0, 0, l
    # Has Child
    if elem['type'] == '[':
        elem2 = l[0]
        if elem2['type'] == '[':
            elem2 = {'token':'','type':'n'}
        elif elem2['type'] == ']':
            elem = {'msg':"Unknown grammar token \"[]\"",'loc':elem['index']}
        else:
            l.pop(0)
        elem = {'token':elem2['token'],'type':elem2['type'],'index':elem['index']}
        move = 'c'
    # Go to parent without inputting a value
    elif elem['type'] == ']':
        elem = 0
        move = 'P'
    else:
        elem = {'token':elem['token'],'type':elem['type'],'index':elem['index']}
        elem2 = l[0]
        if elem2['type'] == ']':
            move = 'p'
            l.pop(0)
        else:
            move = 's'
    return  elem, move, l

##  ##
def Tree2Linear(T,f = None): # Breadth-first traversal
# Convert the tree to a linear format by saving the element and
# number of children. It outputs as a breadth-first traversal. The
# function is designed as a convenient format to save the tree as a file data.
    queue = [T.GetIter()]
    L = []
    if f is None:
        while bool(queue):
            iter = queue.pop(0)
            L.append({'elem':iter.GetElem(),'num':iter.NumChild()})
            for i in range(0,iter.NumChild()):
                queue.append(iter.GetChildIter(i))
    else:
        while bool(queue):
            iter = queue.pop(0)
            L.append({'elem':f(iter.GetElem()),'num':iter.NumChild()})
            for i in range(0,iter.NumChild()):
                queue.append(iter.GetChildIter(i))
    return L


def Linear2Tree(L,f = None):
# Converts the linear formatted tree to an actual tree.
    n = 0
    if f is None:
        T = Tree.Tree(L[n]['elem'])
    else:
        T = Tree.Tree(f(L[n]['elem']))
    queue = [{'iter':T.GetIter(),'num':L[n]['num']}]
    n += 1
    if f is None:
        while n < len(L):
            e = queue.pop(0)
            e['iter'].SetChildElems([L[i]['elem'] for i in range(n,n+e['num'])])
            for i in range(0,e['num']):
                queue.append({'iter':e['iter'].GetChildIter(i),'num':L[n+i]['num']})
            n += e['num']
    else:
            e = queue.pop(0)
            e['iter'].SetChildElems([f(L[i]['elem']) for i in range(n,n+e['num'])])
            for i in range(0,e['num']):
                queue.append({'iter':e['iter'].GetChildIter(i),'num':L[n+i]['num']})
            n += e['num']
    return T

def SaveTrees4Matlab(T,name = 'LinTree.mat'):
# Package for Matlab to check.
    LL = [useful.listfunc(useful.None2neg1,Tree2Linear(t)) for t in T]
    scipy.io.savemat(name,{'LL':LL})
