import copy

## Mutable Tree data structure
class Tree:
    def __init__(self,e):
        self.root = TreeNode(e)
    def GetIter(self):
        # Fetches a TreeIter to the root node
        self.AdjustRoot()
        return TreeIter(self.root)
    def GetElem(self):
        # Fetches the root element
        self.AdjustRoot()
        return self.root.GetElem()
    def PushRoot(self,e):
        # Reinterprets element "e" as the root node of the current root node.
        old_root = self.root
        self.root = TreeNode(e)
        old_root.parent = TreeIter(self.root)
        self.root.child.append(old_root)
    def AdjustRoot(self):
        # When functions such as PushRoot, parameter self.root may no longer be
        # a root node. This function adjusts the self.root node to point to the
        # root node.
        while self.root.hasParent():
            self.root = self.root.parent.node
    def __eq__(self,other):
        match, _, _ = CompareTree(self,other)
        return match
    def __ne__(self,other):
        return (not self.__eq__(other))

# The node container for the Tree data structure
class TreeNode:
    def __init__(self,e = None,p = None):
        self.parent = p
        self.elem = e
        self.child = []
    def NumChild(self):
        return len(self.child)
    def InsertChild(self,e,childIdx,newchildIdx):
    	# Insert element "e" as the new "childIdx"-th node of node
    	# "self". The original "childIdx"-th is changed to be the
    	# "newchildIdx"-th child of the node with element "e".
    	# Returns the TreeIter for the new node.
        newNode = TreeNode(e,TreeIter(self))
        childNode = self.child[childIdx]
        iter = TreeIter(newNode)
        if not(newchildIdx is None):
            newNode.child.extend([TreeNode(None,iter) for i in range(0,newchildIdx)])
        newNode.child.append(childNode)
        childNode.parent = iter
        self.child[childIdx] = newNode
        return iter
    def AddChild(self,e = None):
        # Adds an element "e" as the last child element of the current node.
        self.child.append(TreeNode(e,TreeIter(self)))
    def GetElem(self):
        # Gets the element of the node.
        return self.elem
    def SetElem(self,e):
        # Sets the element of the node.
        self.elem = e
    
    # Children
    def RemoveChild(self,k):
        # Removes the "k"-th child.
        self.child.pop(k)
    def GetChild(self,k):
        # Gets the node to the "k"-th child of the current node.
        if 0 <= k and k < self.NumChild():
            return self.child[k]
        else:
            raise Exception("Attempt to access " + str(k) + "-th child failed.")
    def GetChildElems(self):
        # Gets all the child elements.
        return [self.child[i].elem for i in range(0,self.NumChild())]
    def SetChildElems(self,c):
        # Resets the children of the current node to contain the
        # elements from "c". All original children (and their children)
        # will be erased.
        self.child = [TreeNode(e,TreeIter(self)) for e in c]
    
    # Parent
    def GetParentIter(self):
        # Gets an iterator to the parent node of the current node.
        return self.parent
    def GetParent(self):
        # Gets an element of the parent node of the current node.
        if self.hasParent():
            return self.parent.GetElem()
        else:
            raise Exception(
                    "Can't get parent because the current node has no parent.")
    def InsertParent(self,e,childIdx = 0):
        # Insert element "e" as the new parent node of node "self". The
        # The new parent node is assigned to be the child of the
        # original parent with the same child index as node "self".
        # The node "self" is reinterpreted as the "childIdx"-th child
        # of the new parent node.
        # Returns the TreeIter for the new node.
        iterp = self.GetParentIter()
        newNode = TreeNode(e,iterp)
        itern = TreeIter(newNode)
        newNode.child.extend([TreeNode(None,itern) for i in range(0,childIdx)])
        newNode.child.append(self)
        self.parent = itern
        if not(iterp is None):
            childNumParent = iterp.node.child.index(self)
            iterp.AttachAsChildReplacement(itern,childNumParent)
        return itern
    def hasParent(self):
        return not(self.parent is None)
    
    # Attach as child
    def AttachAsChild(self,node,k):
        # Attaches the "node" as the "k"-th child
        self.child.insert(k,node)
        node.parent = TreeIter(self)
    def AttachAsChildReplacement(self,node,k):
        # Replaces the current "k"-th child to be of value "node"
        self.child[k] = node
        node.parent = TreeIter(self)
    def AttachAsLastChild(self,node):
        # Attaches the "node" as the last child
        self.child.append(node)
        node.parent = TreeIter(self)
    
    # Display Text
    def __str__(self):
        return self.textRep()
    def __repr__(self):
        return self.textRep()
    def textRep(self):
        return "(Element: " + self.elem.__str__() \
            + ", Number of Children: " + str(len(self.child)) \
            + ", Has Parent: " + str(not(self.parent is None)) + ")"


class TreeIter:
    def __init__(self,obj):
        # Creates a tree iterator to the tree node "obj".
        if isinstance(obj,TreeIter):
            self.node = obj.node
        else:
            self.node = obj
    def __eq__(self,other):
        # Compares the elements of the current node and the "other" node.
        return (self.node == other.node)
    
    # Get/Set Elem
    def GetElem(self):
        # Gets the element of the node in the current iterator.
        return self.node.GetElem()
    def SetElem(self,e):
        # Sets the element of the node in the current iterator.
        self.node.SetElem(e)
    
    # Children
    def NumChild(self):
        # Gets the element of child nodes.
        return self.node.NumChild()
    def InsertChild(self,e,childIdx,newchildIdx):
        # Insert element "e" as the new "childIdx"-th node of node of
        # iterator "self". The original "childIdx"-th is changed to be
        # the "newchildIdx"-th child of the node with element "e".
        # Returns the TreeIter for the new node.
        return self.node.InsertChild(e,childIdx,newchildIdx)
    def AddChild(self,e = None):
        # Adds an element "e" as the last child element of the node.
        self.node.AddChild(e)
    def ChildrenMakeRoom(self,num): # num is the new number of children nodes
        # Consider when there is only 2 children present for the current node
        # but one wants to add a 5-th child. Then more children must be
        # constructed before insertion can occur. This function adds more
        # children until the current node has "num" children.
        for i in range(self.NumChild(),num):
            self.AddChild()
    def RemoveChild(self,k):
        # Removes the "k"-th child.
        self.node.RemoveChild(k)
    def ReplaceChild(self,k,iter):
        # Replaces the "k"-th child node with the node of iterator "iter".
        self.node.child[k] = iter.node
    def GetChildIter(self,k = None):
        # Gets an iterator to the "k"-th child node of the current node.
        if k is None:
            return TreeIter(self.node.GetChild(self.node.NumChild()-1))
        else:
            return TreeIter(self.node.GetChild(k))
    def GetChildElems(self):
        # Gets the "k"-th child element.
        return self.node.GetChildElems()
    def SetChildElems(self,c):
        # Resets the children of the current node to contain the
        # elements from "c". All original children (and their children)
        # will be erased.
        self.node.SetChildElems(c)
    
    # Parent
    def InsertParent(self,e,childIdx = None):
        # Insert element "e" as the new parent node of node "self". The
        # The new parent node is assigned to be the child of the
        # original parent with the same child index as node "self".
        # The node "self" is reinterpreted as the "childIdx"-th child
        # of the new parent node.
        # Returns the TreeIter for the new node.
        return self.node.InsertParent(e,childIdx)
    def GetParentIter(self):
        # Gets an iterator to the parent node of the current node.
        return self.node.GetParentIter()
    def hasParent(self):
        return self.node.hasParent()
    
    # Attach as child
    def AttachAsChild(self,iterC,k):
        # Attaches the "node" as the "k"-th child
        self.node.AttachAsChild(iterC.node,k)
    def AttachAsChildReplacement(self,iterC,k):
        # Replaces the current "k"-th child to be of value "node"
        self.node.AttachAsChildReplacement(iterC.node,k)
    def AttachAsLastChild(self,iterC):
        # Attaches the "node" as the last child
        self.node.AttachAsLastChild(iterC.node)
    def __str__(self):
        return self.node.textRep()
    def __repr__(self):
        return self.node.textRep()


## Immutable Tree data structure
class ITree:
    def __init__(self,T):
        # Copy tree "T" to "self"
        iterT = T.GetIter()
        self.root = ITreeNode(iterT.GetElem(),None)
        iterS = self.GetIter()
        stackT = [iterT]
        stackS = [iterS]
        while bool(stackT):
            iterT = stackT.pop()
            iterS = stackS.pop()
            if iterT.NumChild() > 0:
                iterS.SetChildElems(iterT.GetChildElems())
                stackT.extend([iterT.GetChildIter(i)
                                        for i in range(0,iterT.NumChild())])
                stackS.extend([iterS.GetChildIter(i)
                                        for i in range(0,iterT.NumChild())])
    def GetIter(self):
        # Fetches a TreeIter to the root node
        return TreeIter(self.root)
    def __eq__(self,other):
        match, _, _ = CompareTree(self,other)
        return match
    def __ne__(self,other):
        return (not self.__eq__(other))

class ITreeNode:
    def __init__(self,e,p,nodes = []):
        self.parent = p
        self.elem = e
        if bool(nodes) and isinstance(nodes,list):
            self.child = tuple(nodes)
        else:
            self.child = nodes
    def NumChild(self):
        # Gets the element of child nodes.
        return len(self.child)
    def GetElem(self):
        # Gets the element of the node in the current iterator.
        return self.elem
    
    # Children
    def GetChild(self,k):
        # Gets the node to the "k"-th child of the current node.
        if 0 <= k and k < self.NumChild():
            return self.child[k]
        else:
            raise Exception("Attempt to access " + str(k) + "-th child failed.")
    def GetChildElems(self):
        # Gets all the child elements.
        return [self.child[i].elem for i in range(0,len(self.child))]
    
    # Parent
    def GetParentIter(self):
        # Gets an iterator to the parent node of the current node.
        return self.parent
    def GetParent(self):
        # Gets an element of the parent node of the current node.
        if self.hasParent():
            return self.parent.GetElem()
        else:
            raise Exception(
                    "Can't get parent because the current node has no parent.")
    def hasParent(self):
        return not(self.parent is None)
    def SetChildElems(self,c):
        # Sets the children of the current node to contain the
        # elements from "c". All original children (and their children)
        # will be erased.
        # Once the child elements are set, it cannot be reset.
        if isinstance(self.child,list):
            self.child = tuple([TreeNode(e,TreeIter(self)) for e in c])
        else:
            raise Exception("Can't change immutable node.")
    
    # Display Text
    def __str__(self):
        return self.textRep()
    def __repr__(self):
        return self.textRep()
    def textRep(self):
        return "(Element: " + self.elem.__str__() \
            + ", Number of Children: " + str(len(self.child)) \
            + ", Has Parent: " + str(not(self.parent is None)) + ")"


# Tree functions
def CompareTree(T,O,type = 'breadth'):
    # Compares the trees "T" and "O" for any differences through tree
    # traversal of type "type".
    # Returns whether the two trees match and if there is a mismatch,
    # the first TreeIter location of the mismatch from each of the
    # trees.
    iterT = T.GetIter(); iterO = O.GetIter()
    match = True
    containerT = [iterT]
    containerO = [iterO]
    if type == 'breadth':
        while bool(containerT):
            iterT = containerT.pop(0); iterO = containerO.pop(0)
            if (iterT.NumChild() != iterO.NumChild()) or (iterT.GetElem() != iterO.GetElem()):
                match = False
                break
            else:
                for i in range(0,iterT.NumChild()):
                    containerT.append(iterT.GetChildIter(i))
                    containerO.append(iterO.GetChildIter(i))
    elif type == 'depth':
        while bool(containerT):
            iterT = containerT.pop(); iterO = containerO.pop()
            if (iterT.NumChild() != iterO.NumChild()) or (iterT.GetElem() != iterO.GetElem()):
                match = False
                break
            else:
                for i in range(iterT.NumChild()-1,-1,-1):
                    containerT.append(iterT.GetChildIter(i))
                    containerO.append(iterO.GetChildIter(i))
    else:
        raise Exception("Traversal type must be either breadth or depth.")
    return match, iterT, iterO


def TreeConstruct(f,data0):
    # Constructs a tree and returns tree from function "f" with initial data
    # "data0". The construction is of depth-first tree traversal.
    # Function "f" must be of the form [elem,move,data] = f(data) where
    # "move" is equal to 'c','p','P','s',0.
    # Move = 'c' - Go to child (and assign element)
    #        'p' - Go to parent and assign element
    #        'P' - Go to parent without assigning element
    #        's' - Construct sibling, assign element and go to it
    #         0  - End the tree construction
    elem, _, data = f(data0)
    
    # Obtain the root node
    if elem['type'] == '[' or elem['type'] == ']':
        T = Tree()
        err = "Must start with string token"
        return T, data, err
    else:
        T = Tree(elem)
        err = ""
        iter = T.GetIter()
        stack = []
    while True:
        elem, move, data = f(data)
        
        # A switch-case statement would be better:
        if move == 'c': # Go to child
            stack.append(iter)
            iter.AddChild()
            iter = iter.GetChildIter()
            iter.SetElem(elem)
        elif move == 'p': # Go to parent
            if (not stack):
                err = "Has passed the root node"
                break
            # Go to sibling
            iter = stack[-1] # Get top element
            iter.AddChild()
            iter = iter.GetChildIter()
            # Input element
            iter.SetElem(elem)
            # Move to parent
            iter = stack.pop()
        elif move == 'P': # Go to parent without inputting
            if (not stack):
                err = "Has passed the root node"
                break
            # Move to parent
            iter = stack.pop()
        elif move == 's': # Construct sibling
            # Has parent
            if (not stack):
                err = "Cannot be a sibling of a root node"
                break
            # Go to sibling
            iter = stack[-1]
            iter.AddChild()
            iter = iter.GetChildIter()
            # Input element
            iter.SetElem(elem)
        elif move == 0:
            break
        else:
            raise Exception("Unknown tree construction traversal command " + str(move) + ".")
    return T, data, err


def TreeCopy(T,iter = None):
    # Creates a copy of the tree "T". The caller may specify one or a
    # list of TreeIters, "iter", to find a corresponding set of
    # TreeIters for the copy tree.
    iterT = T.GetIter()
    if (iter is None):
        iter = iterT
        
    O = Tree(iterT.GetElem()); iterO = O.GetIter()
    stackT = [iterT]
    stackO = [iterO]
    if isinstance(iter,list):
        iter2 = [None]*len(iter)
        while bool(stackT):
            iterT = stackT.pop()
            iterO = stackO.pop()
            if iterT.NumChild() > 0:
                iterO.SetChildElems(iterT.GetChildElems())
                stackT.extend([iterT.GetChildIter(i) for i in range(0,iterT.NumChild())])
                stackO.extend([iterO.GetChildIter(i) for i in range(0,iterO.NumChild())])
            # Find instances of iterT in list variable: iter
            match = [i for i,x in enumerate(iter) if x == iterT]
            for i in match:
                iter2[i] = iterO
    else:
        iter2 = None
        while bool(stackT):
            iterT = stackT.pop()
            iterO = stackO.pop()
            if iterT.NumChild() > 0:
                iterO.SetChildElems(iterT.GetChildElems())
                stackT.extend([iterT.GetChildIter(i) for i in range(0,iterT.NumChild())])
                stackO.extend([iterO.GetChildIter(i) for i in range(0,iterO.NumChild())])
            if iter == iterT:
                iter2 = iterO
    return O, iter2


def TreeElems(iter,l):
    l.append({'Elem:' : iter.GetElem(), 'numChild' : iter.NumChild()})


def treefun(f,*arg):
    # "arg" is a variable number of trees with the same dimensions. A
    # new tree, with the same dimension, is constructed with elements
    # being f(~), where ~ are the elements for the trees in "arg",
    # corresponding to the same dimension as said element.
    # The new tree is returned.
    iters = [T.GetIter() for T in arg]
    elem = [iter.GetElem() for iter in iters]
    O = Tree(f(*elem)); iterO = O.GetIter()
    stackT = [iters]
    stackO = [iterO]

    while bool(stackT):
        iters = stackT.pop()
        iterO = stackO.pop()
        if iters[0].NumChild() > 0:
            c = [iter.GetChildElems() for iter in iters]
            output = [None]*len(c[0])
            for i in range(0,len(c[0])):
                pack = [e[i] for e in c]
                output[i] = f(*pack)
            iterO.SetChildElems(output)
            for i in range(0,iters[0].NumChild()):
                stackT.append([iter.GetChildIter(i) for iter in iters])
                stackO.append(iterO.GetChildIter(i))
    return O


def TreeTraverse(T, f, data0, type, earlyEnd = False):
    # Traverses of tree traversal type "type" while collecting data
    # with parameters "f" and "data".
    # If type is of "breadth" or "depth", function "f" is called by
    # "f(iter,data)", where "iter" is a TreeIter to the tree node and
    # "data" is the data.
    # If type is of "walk", function "f" is called by expression
    # "f(iter,data,curdir)". "iter" and "data" are the same as above
    # and "curdir" is equal to
    #       'd' - About to go down
    #       'u' - About to go up
    #       'e' - About to end traversal
    # If earlyEnd is True, then f is expected to output two parameters:
    # data and endEarly. Output data is the data collected and to be inputted.
    # Output endEarly is a boolean to determine if the traversal should end
    # at the current tree node.
    # If earlyEnd is False, then f is expected to output only data.
    data = copy.deepcopy(data0)
    container = [T.GetIter()]
    if earlyEnd:
        if type == 'depth':
            while bool(container):
                iter = container.pop()
                data,endEarly = f(iter,data)
                if endEarly:
                    break
                for i in range(iter.NumChild()-1,-1,-1):
                    container.append(iter.GetChildIter(i))
        elif type == 'breadth':
            while bool(container):
                iter = container.pop(0)
                data,endEarly = f(iter,data)
                if endEarly:
                    break
                for i in range(0,iter.NumChild()):
                    container.append(iter.GetChildIter(i))
        elif type == 'walk':
            stack = [0]
            iter = T.GetIter()
            while True:
                index = stack.pop()
                if iter.NumChild() > index:
                    data,endEarly = f(iter,data,'d')
                    if endEarly:
                        break
                    stack.append(index + 1)
                    iter = iter.GetChildIter(index)
                    stack.append(0)
                elif (not stack):
                    data,endEarly = f(iter,data,'e')
                    break
                else:
                    data,endEarly = f(iter,data,'u')
                    if endEarly:
                        break
                    iter = iter.GetParentIter()
        else:
            raise Exception("Traversal type must be either breadth, depth or walk.")
    else:
        if type == 'depth':
            while bool(container):
                iter = container.pop()
                data = f(iter,data)
                for i in range(iter.NumChild()-1,-1,-1):
                    container.append(iter.GetChildIter(i))
        elif type == 'breadth':
            while bool(container):
                iter = container.pop(0)
                data = f(iter,data)
                for i in range(0,iter.NumChild()):
                    container.append(iter.GetChildIter(i))
        elif type == 'walk':
            stack = [0]
            iter = T.GetIter()
            while True:
                index = stack.pop()
                if iter.NumChild() > index:
                    data = f(iter,data,'d')
                    stack.append(index + 1)
                    iter = iter.GetChildIter(index)
                    stack.append(0)
                elif (not stack):
                    data = f(iter,data,'e')
                    break
                else:
                    data = f(iter,data,'u')
                    iter = iter.GetParentIter()
        else:
            raise Exception("Traversal type must be either breadth, depth or walk.")
    return data

def TreeIndex(T,type = 'breadth'):
    # Creates a new tree of same dimension as "T" with each of the tree
    # node having a zero-based index.
    iterT = T.GetIter()
    n = 0
    O = Tree(n); iterO = O.GetIter()
    containerT = [iterT]
    containerO = [iterO]
    if type == 'breadth':
        while bool(containerT):
            iterT = containerT.pop(0)
            iterO = containerO.pop(0)
            if iterT.NumChild() > 0:
                iterO.SetChildElems(list(range(n+1,n+1+iterT.NumChild())))
                n = n + iterT.NumChild()
                for i in range(0,iterT.NumChild()):
                    containerT.append(iterT.GetChildIter(i))
                    containerO.append(iterO.GetChildIter(i))
    elif type == 'depth':
        while bool(containerT):
            iterT = containerT.pop()
            iterO = containerO.pop()
            if iterT.NumChild() > 0:
                iterO.SetChildElems(list(range(n+1,n+1+iterT.NumChild())))
                n = n + iterT.NumChild()
                for i in range(0,iterT.NumChild()):
                    containerT.append(iterT.GetChildIter(i))
                    containerO.append(iterO.GetChildIter(i))
    else:
        raise Exception("Traversal type must be either breadth or depth.")
    return O


def TreePropagate(T,f,data0,type = 'depth'):
    # [data] = f(iter,data).
    data = copy.deepcopy(data0)
    l = []
    container = [{'iter' : T.GetIter(), 'data' : data}]
    if type == 'depth':
        while bool(container):
            s = container.pop()
            data = f(s['iter'],s['data'])
            if s['iter'].NumChild() > 0:
                for i in range(0,s['iter'].NumChild()):
                    container.append({'iter' : s['iter'].GetChildIter(i), 'data' : s['data']})
            else:
                l.append(s['data'])
    elif type == 'breadth':
        while bool(container):
            s = container.pop(0)
            data = f(s['iter'],s['data'])
            if s['iter'].NumChild() > 0:
                for i in range(0,s['iter'].NumChild()):
                    container.append({'iter' : s['iter'].GetChildIter(i), 'data' : s['data']})
            else:
                l.append(s['data'])
    else:
        raise Exception("Traversal type must be either breadth or depth.")
    return l


def TreeSearch(T,elem,f = (lambda x,y : x == y),type = 'depth'):
    # Find the TreeIters representing the tree nodes of tree "T" with
    # elements matching "elem". The comparison is performed using
    # function "f" in the tree traversal of type "type".
    data = {'Elem' : elem, 'l' : [], 'f' : f}
    data = TreeTraverse(T,TreeSearchTraverse,data,type);
    return data['l']


def TreeSearchTraverse(iter,data0):
    data = copy.deepcopy(data0)
    if data['f'](iter,data):
        data['l'].append(iter)
    return data