from heapq import heapify, heappop, heappush

class NodArbore:
    def __init__(self, _informatie, _parinte=None, _g = 0, _h = 0):
        self.informatie = _informatie
        self.parinte = _parinte
        self.g = _g
        self.h = _h
        self.f = _g + _h

    def __lt__(self, elem):
        return self.f < elem.f or (self.f == elem.f and self.h < elem.h)

    def drumRadacina(self):
        nod = self
        l = []
        while nod:
            l.append(nod.informatie)
            nod = nod.parinte
        return l[::-1]

    def inDrum(self, infoNod):
        nod = self
        while nod:
            if nod.informatie == infoNod:
                return True
            nod = nod.parinte
        return False

    def __str__(self):
        return str(self.informatie)

    def __repr__(self):
        sirDrum = "->".join(map(str, self.drumRadacina()))
        return f"{self.informatie}, cost: {self.g}, ({sirDrum})"

#######################

class Graf:
    def __init__(self, _matr, _start, _scopuri, estimations):
        self.matr = _matr
        self.start = _start
        self.scopuri = _scopuri
        self.h = estimations

    def scop(self, informatieNod):
        return informatieNod in self.scopuri
    
    def validate(self):
        return True
    
    def get_estimation(self, node_idx):
        return self.h[node_idx]
    
    def get_move_cost(self, src, dest):
        return self.matr[src][dest]

    def succesori(self, nod):
        lSuccesori = []
        for infoSuccesor in range(len(self.matr)):
            conditieMuchie = self.matr[nod.informatie][infoSuccesor] != 0
            conditieNotInDrum = not nod.inDrum(infoSuccesor)
            if conditieMuchie and conditieNotInDrum:
                nodNou = NodArbore(infoSuccesor, nod, nod.g + self.get_move_cost(nod.informatie, infoSuccesor), self.get_estimation(infoSuccesor))
                lSuccesori.append(nodNou)
        return lSuccesori

#######################

def A_star(gr, sol_count):
    open = [NodArbore(gr.start)]
    heapify(open)

    processed_cost = {}
    processed_cost[open[0].informatie] = 0

    while open and sol_count > 0:
        nodCurent = heappop(open)
        
        if gr.scop(nodCurent.informatie):
            print("Solutie: ", end="")
            print(repr(nodCurent))
            sol_count -= 1
            continue

        if processed_cost[nodCurent.informatie] < nodCurent.f:
            continue

        successors = gr.succesori(nodCurent)

        for s in successors:
            if processed_cost.get(s.informatie, False) and processed_cost[s.informatie] <= s.f:
                continue

            processed_cost[s.informatie] = s.f

            # open = [elem for elem in open if elem.informatie != s.informatie] # comment for multiple solutions
            heappush(open, s)


m = [
    [0, 3, 5, 10, 0, 0, 100],
    [0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 4, 9, 3, 0],
    [0, 3, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 0, 5],
    [0, 0, 3, 0, 0, 0, 0],
]

start = 0
scopuri = [4,6]
h=[0,1,6,2,0,3,0]
gr = Graf(m, start, scopuri, h)

if gr.validate():
    A_star(gr, 3)
else:
    print("Invalid start state")