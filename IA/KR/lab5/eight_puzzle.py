import copy
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
    def __init__(self, _start, _scopuri):
        self.start = _start
        self.scopuri = _scopuri

    def scop(self, informatieNod):
        return informatieNod in self.scopuri
    
    def validate(self):
        inversions_count = 0
        spread_matrix = self.start[0] + self.start[1] + self.start[2]
        for i, elem1 in enumerate(spread_matrix[:-1]):
            for j, elem2 in enumerate(spread_matrix[i + 1:]):
                if elem1 * elem2 != 0:
                    inversions_count += elem1 > elem2

        return 1 - (inversions_count % 2)
    
    def manhattan_dist(self, r1, c1, r2, c2):
        return abs(r2 - r1) + abs(c2 - c1)
    
    def get_estimation(self, infoNod):
        dist = 0

        for i, line in enumerate(infoNod):
            for j, elem in enumerate(line):
                if elem == 0:
                    continue

                real_line = (elem - 1) // 3
                real_col = (elem - 1) % 3
                dist += self.manhattan_dist(i, j, real_line, real_col)

        return dist
    
    def get_move_cost(self, src, dest):
        return 1

    def succesori(self, nod):
        def find_goal(matr):
            for i, linie in enumerate(matr):
                for j, elem in enumerate(linie):
                    if elem == 0:
                        return i, j
                    
        lSuccesori = []
        goal_line, goal_col = find_goal(nod.informatie)
        dir = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        
        for d in dir:
            new_line = goal_line + d[0]
            new_col = goal_col + d[1]

            if new_line >= 0 and new_line < 3 and new_col >= 0 and new_col < 3:
                infoSuccesor = copy.deepcopy(nod.informatie)
                infoSuccesor[goal_line][goal_col], infoSuccesor[new_line][new_col] = infoSuccesor[new_line][new_col], infoSuccesor[goal_line][goal_col]

                if not nod.inDrum(infoSuccesor):
                    nodNou = NodArbore(infoSuccesor, nod, nod.g + 1, self.get_estimation(infoSuccesor))
                    lSuccesori.append(nodNou)

        return lSuccesori

#######################

def make_hashable(obj):
    if isinstance(obj, list):
        return tuple(make_hashable(x) for x in obj)
    elif isinstance(obj, dict):
        return tuple((k, make_hashable(v)) for k, v in sorted(obj.items()))
    return obj

def A_star(gr, sol_count):
    open = [NodArbore(gr.start)]
    heapify(open)

    processed_cost = {}
    processed_cost[make_hashable(open[0].informatie)] = 0

    while open and sol_count > 0:
        nodCurent = heappop(open)
        
        if gr.scop(nodCurent.informatie):
            print("Solutie: ", end="")
            print(repr(nodCurent))
            sol_count -= 1
            continue

        if processed_cost[make_hashable(nodCurent.informatie)] < nodCurent.f:
            continue

        successors = gr.succesori(nodCurent)

        for s in successors:
            if processed_cost.get(make_hashable(s.informatie), False) and processed_cost[make_hashable(s.informatie)] <= s.f:
                continue

            processed_cost[make_hashable(s.informatie)] = s.f

            # open = [elem for elem in open if elem.informatie != s.informatie] # comment for multiple solutions
            heappush(open, s)

def aStar(gr):
    OPEN = [NodArbore(gr.start)]
    CLOSED=[]
    while OPEN:
        nodCurent = OPEN.pop(0)
        CLOSED.append(nodCurent)
        if gr.scop(nodCurent.informatie):
            print("Solutie: ", end="")
            print(repr(nodCurent))
            return
        lSuccesori=gr.succesori(nodCurent)
        gasitOPEN=False
        for s in lSuccesori:
            for nod in OPEN:
                if s.informatie==nod.informatie:
                    gasitOPEN=True
                    if s<nod:
                        OPEN.remove(nod)
                    else:
                        lSuccesori.remove(s)
                    break
            if not gasitOPEN:
                for nod in CLOSED:
                    if s.informatie==nod.informatie:
                        if s<nod:
                            CLOSED.remove(nod)
                        else:
                            lSuccesori.remove(s)
                        break
        OPEN+=lSuccesori
        OPEN.sort()  
    print("Nu avem solutii")

f=open("input.txt","r")
start = [list(map(int, linie.strip().split())) for linie in f.read().strip().split("\n")]
scopuri=[[[1, 2, 3], [4, 5, 6], [7, 8, 0]]]
f.close()
print(start)
gr = Graf(start, scopuri)

if not gr.validate():
    print("No solutions")
else:
    # A_star(gr, 1)
    aStar(gr)