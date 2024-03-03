def find(l,obj = True,f = (lambda x,y : x == y)):
    # Finds object "obj" in the list of elements "l" and returns the
    # index location of the match. Comparison is performed using optional
    # function "f".
    return [p for p,e in enumerate(l) if f(e,obj)]


def select(l,b,type = None):
    # Selects the elements from list "l" using a list of booleans of the same
    # length or a set of integer indices.
    # For example, l = [0,1,2,3,4,5],b1 = [True,False,False,True,True,False],
    # b2 = [3,0,4,4].
    # Then [0,3,4] = select(l,b1) and [3,0,4,4] = select(l,b2).
    # Parameter "type" is used to interprete force an 'bool' or 'int'
    # interpretation of parameter "b".
    if type == None:
        if isinstance(b[0],bool):
            return [e for i,e in enumerate(l) if b[i]]
        elif isinstance(b[0],int):
            return [l[i] for i in b]
        else:
            raise Exception("Unknown selection type of first type of second parameter.")
    else:
        if type == 'bool':
            return [e for i,e in enumerate(l) if b[i]]
        elif type == 'int':
            return [l[i] for i in b]
        else:
            raise Exception("Unknown selection type " + type + ".")


def unique(l,f = (lambda x,y : x == y)):
    # Returns the unique elements in the list "l". Comparison is performed
    # using optional function "f".
    match = [None]*len(l)
    for i in range(len(l)):
        for j in range(i):
            if f(l[i],l[j]):
                match[i] = j
    newl = [l[i] for i in range(0,len(l)) if match[i] == None]
    return {'l':newl,'match':match}


def CheckDuplicates(c):
    # Check for duplicates. Returns None if there are no duplicates
    # and an index pair of the first mismatching pair found.
    dup = None
    for i in range(0,len(c)):
        for j in range(i+1,len(c)):
            if c[i] != c[j]:
                dup = (i,j)
    return dup


def listfunc(f,*arg):
    # Applies function "f" to the list of parameters "arg". For example,
    # if f = (lambda x,y,z:z*max(x,y)), then
    # [4,-3,-4,5] = listfunc(f,[1,2,4,5],[4,3,2,1],[1,-1,-1,1])
    L = [None]*len(arg[0])
    for i in range(0,len(arg[0])):
        pack = [e[i] for e in arg]
        L[i] = f(*pack)
    return L


def None2Zero(e):
    # Converts value None to 0.
    return 0 if (e is None) else e


def None2neg1(x):
    # Converts value None to -1.
    return -1 if (x is None) else x


def neg12None(x):
    # Converts value -1 to None.
    return -1 if (x == -1) else x

