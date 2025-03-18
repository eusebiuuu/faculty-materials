
class NodArbore:
    def __init__(self, _informatie, _parinte = None):
        self.informatie = _informatie
        self.parinte = _parinte

    def drumRadacina(self):
        nod = self
        l = []
        while nod:
            l.append(nod)
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
        sirDrum = "->".join([str(nod) for nod in self.drumRadacina()])
        return f"{self.informatie}: ({sirDrum})"



class Graf:
    def __init__(self, _start, _scopuri):
        self.start = _start
        self.scopuri = _scopuri

    def scop(self, informatieNod):
        return informatieNod in self.scopuri
    

    '''

        - i = canibali
        - j = misionari

        state: (i, j, k)

        current state: (i, j, 0)
        if k = 1: (n - i, n - j, 1)

        1. i > j
            1. j = 0 -> (i', 0, 1 - k), i' = (i - min(m, i)), (i - 1)
            2. j > 0 -> -1
        2. i < j:
            1. i = 0 -> (0, j', 1 - k), j' = (j - min(m, j)), (j - 1)
            2. j = n
                justificare: {(a, b) a.i. a <= b && i - a <= n - b && n - i + a <= b <=> n - b <= i - a}
                tranzitie: (i - a, n - b, 1 - k), a = 0 while n - i + 2 * a <= m && a <= i
            3. j < n && i < j <=> n - i > n - j > 0 -> -1
        3. i = j:
            1. i = 0 -> k = 0 ? -1 : consider answer
            2. i > 0 -> (i', i', 1 - k), i' = (i - min(m / 2, i)), (i - 1)

        next state: (i', j', 1)
        if k = 1: (n - i', n - j', 0)

    '''

    def succesori(self, current_node):
        successors = []

        def get_next_state(state, boat_place):
            if boat_place == 0:
                return state
            return (Graf.N - state[0], Graf.N - state[1], 0)

        def add_to_list(state, boat_place):
            successors.append(NodArbore(get_next_state(state, boat_place), current_node))

        current_state = current_node.informatie

        if current_state[2] == 1:
            current_state[0] = Graf.N - current_state[0]
            current_state[1] = Graf.N - current_state[1]

        cannibal_count = current_state[0]
        missionary_count = current_state[1]
        boat = current_state[2]

        next_state = (0, 0, boat)

        if cannibal_count > missionary_count:
            if missionary_count == 0:
                for diff in range(1, min(Graf.M, cannibal_count) + 1):
                    next_state[0] = cannibal_count - diff
                    next_state[1] = 0
                    next_state[2] ^= 1
                    add_to_list(next_state, boat)
        elif cannibal_count < missionary_count:
            if cannibal_count == 0:
                for diff in range(1, min(Graf.M, missionary_count) + 1):
                    next_state[0] = 0
                    next_state[1] = missionary_count - diff
                    next_state[2] ^= 1
                    add_to_list(next_state, boat)
            else:
                if missionary_count == Graf.N:
                    cannibal_take = 0
                    while Graf.N - cannibal_count + 2 * cannibal_take <= Graf.M and cannibal_take <= cannibal_count:
                        next_state[0] = cannibal_count - cannibal_take
                        next_state[1] = cannibal_count - cannibal_take
                        next_state[2] ^= 1
                        add_to_list(next_state, boat)
                        cannibal_take += 1
        else:
            if cannibal_count > 0:
                for diff in range(1, min(Graf.M // 2, cannibal_count) + 1):
                    next_state[0] = cannibal_count - diff
                    next_state[1] = cannibal_count - diff
                    next_state[2] ^= 1
                    add_to_list(next_state, boat)
            
        return successors


def BF(gr, nsol):
    coada = [NodArbore(gr.start)]

    while coada and nsol > 0:
        nodCurent = coada.pop()

        for next_node in gr.succesori(nodCurent):
            if gr.scop(next_node.informatie):
                print("Solutie: ", end="")
                print(repr(next_node))
                nsol -= 1
                if nsol == 0:
                    break
            else:
                coada += [next_node]


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

################################

input_file = open("./input.txt", "r")

input = input_file.read().split()

Graf.N = int(input[0])
Graf.M = int(input[1])

input_file.close()

start = (Graf.N, Graf.M, 0)
scopuri = [(0, 0, 1)]
gr=Graf(start, scopuri)
BF(gr, 1)
# DF(NodArbore(start), 4)
# DFN(NodArbore(start), 4)