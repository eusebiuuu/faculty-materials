import copy

class NodArbore:
    def __init__(self, _informatie, _parinte=None, _g=0, _h=0):
        self.informatie = _informatie
        self.parinte = _parinte
        self.g = _g
        self.h = _h
        self.f = _g + _h
        
    def __lt__(self,elem):
        return self.f<elem.f or (self.f==elem.f and self.h<elem.h) 

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
        return f"{self.informatie}, cost:{self.g}, ({sirDrum})"

class Graf:
    def __init__(self,  _start, _scopuri):
        self.start = _start
        self.scopuri = _scopuri
        
    def estimeaza_h(self,infoNod):
        if self.scop(infoNod):
            return 0
        min_estimation = float("inf")
        for goal in self.scopuri:
            curr_estimation = 0
            for i, goal_stack in enumerate(goal):
                for j, block in enumerate(goal_stack):
                    try:
                        curr_estimation += block != infoNod[i][j]
                    except:
                        curr_estimation += 1
            min_estimation = min(min_estimation, curr_estimation)
        return min_estimation

    def scop(self, informatieNod):
        return informatieNod in self.scopuri

    def succesori(self, nod):
        lSuccesori = []

        for i, curr_stack in enumerate(nod.informatie):
            if not curr_stack:
                continue

            temp_state = copy.deepcopy(nod.informatie)
            extracted_block = temp_state[i].pop()

            for j, _ in enumerate(nod.informatie):
                if i == j:
                    continue

                next_state = copy.deepcopy(temp_state)
                next_state[j].append(extracted_block)

                infoSuccesor = next_state

                if not nod.inDrum(infoSuccesor):
                    nodNou = NodArbore(infoSuccesor, nod, nod.g + 1, self.estimeaza_h(infoSuccesor))
                    lSuccesori.append(nodNou)

        return lSuccesori


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


def get_stacks(state):
    return [stack.strip().split() if stack != '#' else [] for stack in state.strip().split('\n')]


f=open("input.txt","r")
start, ends = f.read().split('=========')

start_state = get_stacks(start)

goals = [get_stacks(goal) for goal in ends.strip().split("---")]

f.close()
gr = Graf(start_state, goals)
aStar(gr)
