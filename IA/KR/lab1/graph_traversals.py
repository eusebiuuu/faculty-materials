#inlocuiti fiecare comentariu TODO 

class NodArbore:
    def __init__(self, _informatie, _parinte=None):
        self.informatie = _informatie
        self.parinte = _parinte

    def drumRadacina(self):
        # Return the path from the root to the current node
        nod = self
        l = []
        while nod:
            l.append(nod.informatie)
            nod = nod.parinte
        return l[::-1]

    def inDrum(self, infoNod):
        # Check if a node is in the path from the root to the current node
        nod = self
        while nod:
            if nod.informatie == infoNod:
                return True
            nod = nod.parinte
        return False

    def __str__(self):
        return str(self.informatie)

    def __repr__(self):
        # Return a string representation of the node and its path
        sirDrum = "->".join(map(str, self.drumRadacina()))
        return f"{self.informatie}, ({sirDrum})"

class Graf:
    def __init__(self, _matr, _start, _scopuri):
        self.matr = _matr
        self.start = _start
        self.scopuri = _scopuri

    def scop(self, informatieNod):
        # Check if a node is a goal node
        return informatieNod in self.scopuri

    def succesori(self, nod):
        # Generate successors for a given node
        lSuccesori = []
        for infoSuccesor in range(len(self.matr)):
            conditieMuchie = self.matr[nod.informatie][infoSuccesor] == 1
            conditieNotInDrum = not nod.inDrum(infoSuccesor)
            if conditieMuchie and conditieNotInDrum:
                nodNou = NodArbore(infoSuccesor, nod)
                lSuccesori.append(nodNou)
        return lSuccesori

m = [
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
]

def BF(gr, nsol):
    # Breadth-First Search algorithm
    coada = [NodArbore(gr.start)]
    while coada:
        nodCurent = coada.pop(0)
        if gr.scop(nodCurent.informatie):
            print("Solutie: ", end="")
            print(repr(nodCurent))
            nsol -= 1
            if nsol == 0:
                return
        coada += gr.succesori(nodCurent)

def depth_first(gr, nsol=1):
    # Depth-First Search algorithm
    return DF(NodArbore(gr.start), nsol)

def DF(nodCurent, nsol):
    # Recursive Depth-First Search helper function
    if nsol <= 0:
        return nsol

    if gr.scop(nodCurent.informatie):
        print("Solutie: ", end="")
        print(repr(nodCurent))
        print("\n----------------\n")
        nsol -= 1
        if nsol == 0:
            return nsol

    lSuccesori = gr.succesori(nodCurent)
    for sc in lSuccesori:
        if nsol != 0:
            nsol = DF(sc, nsol)

    return nsol


def DF_Non_Recursive(gr, nsol):
    # Non-Recursive Depth-First Search algorithm
    stack = [NodArbore(gr.start)]
    while stack:
        nodCurent = stack.pop()
        if gr.scop(nodCurent.informatie):
            print("Solutie: ", end="")
            print(repr(nodCurent))
            nsol -= 1
            if nsol == 0:
                return
        stack.extend(gr.succesori(nodCurent)[::-1])

def DF(nodCurent, nsol):
    if nsol <= 0:  
        return nsol

    if gr.scop(nodCurent.informatie):
        print("Solutie: ", end="")
        print(repr(nodCurent.drumRadacina()))
        print("----------------")

        nsol -= 1
        if nsol <= 0:
            return nsol
        
    lSuccesori = gr.succesori(nodCurent)
    for sc in lSuccesori:
        if nsol > 0:
            nsol = DF(sc, nsol)

    return nsol


def DFN(curr_node, nsol):
    print("\n######################\n")
    idx = {}
    in_stack = {}
    in_stack[curr_node] = True

    nodes = [curr_node]

    while len(nodes) > 0:
        curr_node = nodes[-1]

        if gr.scop(curr_node.informatie):
            print("Solutie: ", end="")
            print(repr(nodes))
            print("----------------")
            nsol -= 1

        if nsol == 0:
            break

        curr_successors = gr.succesori(curr_node)
        curr_idx = idx.get(curr_node, -1) + 1
        
        if curr_idx < len(curr_successors):
            next_node = curr_successors[curr_idx]
            idx[curr_node] = curr_idx
            
            if not in_stack.get(next_node, None):
                nodes.append(next_node)
                in_stack[next_node] = True            
        else:
            nodes.pop()
            idx[curr_node] = -1
            in_stack[curr_node] = False

start = 0
scopuri = [5, 9]
gr = Graf(m, start, scopuri)
# BF(gr, 4)
# depth_first(gr, 4)
DF_Non_Recursive(gr, 4)
