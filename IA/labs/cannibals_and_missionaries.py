from queue import Queue


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

    def print_solution(self, file_name, total):
        left = ["(Stanga:<barca>)", "(Stanga)"]
        right = ["(Dreapta)", "(Dreapta:<barca>)"]
        
        src = ["stang", "drept"]
        dest = ["drept", "stang"]
        
        write_file = open(file_name, 'a')
        
        def print_state(cannibals, missionaries, boat):
            write_file.write(f'{left[boat]} {cannibals} canibali {missionaries} misionari  ......  {right[boat]} {total - cannibals} canibali {total - missionaries} misionari\n\n')

        def print_trip(cannibals, missionaries, boat):
            write_file.write(f">>> Barca s-a deplasat de la malul {src[boat]} la malul {dest[boat]} cu {cannibals} canibali si {missionaries} misionari.\n")

        print_state(total, total, 0)

        cannibals = total
        missionaries = total

        path = self.drumRadacina()

        for node in path:
            diff_cannibals = cannibals - node.informatie[0]
            diff_missionaries = missionaries - node.informatie[1]

            if diff_cannibals > 0 or diff_missionaries > 0:
                print_trip(diff_cannibals, diff_missionaries, 1)
            else:
                print_trip(-1 * diff_cannibals, -1 * diff_missionaries, 0)

            print_state(node.informatie[0], node.informatie[1], node.informatie[2])
        
        write_file.close()


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
            1. j = n
                (a, b) a.i. (a <= b || b = 0) && (i - a <= n - b || b = n) && (n - i + a <= b || b = 0)
                1. b = 0 -> (i', n, 1 - k), i' = (i - min(m, i))..(i - 1)
                2. b = n (m >= n) -> (i', 0, 1 - k), i' = (i - min(m - n, i))..(i)
                3. 0 < b < n: (i - a, n - b, 1 - k), a = 0 while n - i + 2 * a <= m && a <= i
            2. i < j < n <=> n - i > n - j > 0 -> -1
        3. i = j:
            1. i = 0 -> k = 0 ? -1 : goal
            2. i > 0
                (a, b) a.i. (i - a <= i - b || b = i) && (a <= b || b = 0) && (n - i + a <= n - i + b || (n - i + b = 0 <=> (i = n && b = 0)))
                
                1. b = 0 => i = n -> (i', n, 1 - k), i' = (i - min(m, i))..(i - 1)
                2. b = i -> m < i ? -1 : (i', 0, 1 - k), i' = (i - min(m - i, i))..(i)
                3. 0 < b < i => a = b -> (i', i', 1 - k), i' = (i - min(m / 2, i))..(i - 1)

        next state: (i', j', 1)
        if k = 1: (n - i', n - j', 0)

    '''

    def succesori(self, current_node):
        successors = []

        def get_next_state(cannibal_count, missionary_count, boat_place):
            if boat_place == 0:
                return (cannibal_count, missionary_count, 1)
            return (Graf.N - cannibal_count, Graf.N - missionary_count, 0)

        def add_to_list(cannibal_count, missionary_count, boat_place):
            next_state = get_next_state(cannibal_count, missionary_count, boat_place)
            if not current_node.inDrum(next_state):
                successors.append(NodArbore(next_state, current_node))

        current_state = current_node.informatie

        cannibal_count = current_state[0]
        missionary_count = current_state[1]
        boat = current_state[2]

        if boat == 1:
            cannibal_count = Graf.N - cannibal_count
            missionary_count = Graf.N - missionary_count

        if cannibal_count > missionary_count:
            if missionary_count == 0:
                for diff in range(1, min(Graf.M, cannibal_count) + 1):
                    add_to_list(cannibal_count - diff, 0, boat)
        elif cannibal_count < missionary_count:
            if missionary_count == Graf.N:
                for diff in range(1, min(Graf.M, cannibal_count) + 1):
                    add_to_list(cannibal_count - diff, Graf.N, boat)

                if Graf.M >= Graf.N:
                    for diff in range(0, min(Graf.M - Graf.N, cannibal_count) + 1):
                        add_to_list(cannibal_count - diff, 0, boat)
                
                cannibal_take = 0
                while Graf.N - cannibal_count + 2 * cannibal_take <= Graf.M and cannibal_take <= cannibal_count:
                    add_to_list(cannibal_count - cannibal_take, cannibal_count - cannibal_take, boat)
                    cannibal_take += 1
        else:
            if cannibal_count == 0:
                return []

            if cannibal_count == Graf.N:
                for diff in range(1, min(Graf.M, cannibal_count) + 1):
                    add_to_list(cannibal_count - diff, Graf.N, boat)
            
            if Graf.M >= cannibal_count:
                for diff in range(0, min(Graf.M - missionary_count, cannibal_count) + 1):
                    add_to_list(cannibal_count - diff, 0, boat)
            
            for diff in range(1, min(Graf.M // 2, cannibal_count) + 1):
                add_to_list(cannibal_count - diff, cannibal_count - diff, boat)
            
        return successors

def BF(gr, nsol):
    coada = Queue()
    coada.put(NodArbore(gr.start))

    while not coada.empty() and nsol > 0:
        current_node = coada.get()

        for next_node in gr.succesori(current_node):
            if gr.scop(next_node.informatie):
                print("Solutie: ", end="")
                print(repr(next_node))

                next_node.print_solution("output.txt", Graf.N)

                nsol -= 1
                if nsol == 0:
                    break
            else:
                coada.put(next_node)


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

start = (Graf.N, Graf.N, 0)
scopuri = [(0, 0, 1)]
gr=Graf(start, scopuri)
BF(gr, 2)
# DF(NodArbore(start), 4)
# DFN(NodArbore(start), 4)