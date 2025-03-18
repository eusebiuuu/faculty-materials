## Notiuni generale
- Programul reprezinta un **agent inteligent**
- In rezolvarea unei probleme agentul realizeaza mai multe etape:
  - **Formularea scopului**
    - determinarea setului de **scopuri** (setului de stari destinatie)
  - **Formularea problemei**
    - determinarea **spatiului starilor** si **actiunilor** (nodurilor si muchiilor arborelui corespunzator problemei) care trebuie luate in considerare
    - definirea **starii initiale** / **starilor initiale**
    - **multimea operatorilor**: reprezinta actiunile ce trebuie considerate pentru a ne muta dintr-o stare oarecare
    - **testul scop**: poate fi o conditie de apartenenta, de verificare etc. a starii curente pentru a apartine multimii de stari scop
    - **functia de scop a unui drum**: suma costurilor (individuale) care compun drumul
  - **[Cautarea](#cautarea)**
  - **Executia**: la final, atunci cand au fost obtinute toate solutiile, se va executa actiunea care ajunge in cea optima si se va relua procesul

## Cautarea
- procesul de decizie al agentului se bazeaza pe examinarea starilor in care poate tranzitiona; acest proces se numeste **cautare**
- cautarea se bazeaza pe o **strategie de cautare**, care trebuie sa indeplineasca **completitudinea**, **complexitatea timpului**, **complexitatea spatiului**, **optimalitatea**
- pentru acest pas se va construi un **arbore de cautare** pentru a determina **drumurile** in **spatiul starilor**
- important de precizat este ca arborele de cautare este format din structuri de date numite **noduri**, iar spatiul starilor este alcatuit din **stari** ce reprezinta configuratii ale mediului; de asemenea, 2 noduri pot stoca informatia aceleiasi stari
- structura de date ce defineste nodurile trebuie sa stocheze urmatoarele informatii: starea referita, parintele, operatorul folosit pentru a se genera nodul, adancimea nodului, costul drumului de la starea initiala pana la nod
- In timpul cautarii, exista mai multe noduri care doresc a fi extinse, ca urmare avem nevoie de o structura de date pentru a realiza selectia celui mai bun nod; de cele mai multe ori va fi folosita o coada
- Cautarile pot fi **neinformate (oarbe)** sau **informate (euristice)**

## Cautarea best-first
- Fie o functie euristica $f$ ce reprezinta rezultatul evaluarii unei stari, numar real, bazat pe informatii specifice domeniului in care s-a formulat problema
- La fiecare pas se va alege nodul cu functia euristica cea mai mica si se va extinde cu toti succesorii lui
- Totusi, in aceasta maniera nu vom lua in considerare drumurile cu costul mai mic dar cu o functie euristica mai mare
- Ca urmare, $f$ va fi formata din suma dintre $g$ si $h$, unde $g(u)$ reprezinta costul minim de la starea initiala la nodul $u$, iar $h(u)$ este functia euristica de mai sus, adica costul minim al unui drum de la $u$ la un nod scop
- In imagine, se va prezenta un algoritm de cautare care poate fi particularizat cu BF, DF sau best-first (deci si cautare informata si neinformata), numit **Graph Search**
- **IMPORTANT**: daca spatiul starilor nu este arbore, atunci pot aparea cicluri; ele trebuie detectate in pasul 6
![](./images/GraphSearch.png)

## A*
- Algoritmul A* este o particularizare a acestuia, cu mentiunile ca acea cautare este de tipul best-first, iar nodurile vor fi ordonate in coada in functie de costurile lor
- De asemenea, $f(n) = g(n) + h(n)$, unde $f(n)$ reprezinta costul minim al unui drum de la start la un nod scop, $g(n)$ este drumul de cost minim de la start la $n$, iar $h(n)$ este costul minim efectiv de la $n$ la un nod scop
- Cum nu putem prezice viitorul, definim o functie $h'(n)$ ce are rolul de a oferi o estimare pentru $h$, $g'$ ca fiind $g$-ul gasit pana la pasul curent din parcurgere si $f'$ distanta minima pana acum de la start la un nod scop
- Ca urmare, avem ca $f'(n) = g'(n) + h'(n)$
![](./images/A*algorithm.png)

## Analiza algoritmului A*
- A* este un algoritm complet, admisibil si optim, 
- Orice algoritm care garantează găsirea unui drum optim la
scop este un **algoritm admisibil**.
- Pentru ca A* sa fie admisibil, avem nevoie de 3 conditii:
  - $h'(n) \le h(n)$
  - orice nod are un numar finit de succesori
  - toate costurile sunt pozitive
- Prin extensie, daca $h'$ nu depaseste pe $h$ atunci $h$ este admisibila

## Aplicatii ale algoritmului A* in Game Theory. Algoritmul minimax
- Incertitudinea din teoria jocurilor, la fel ca in lumea reala, reise din incapacitatea oamenilor sa conceapa o solutie intr-o perioada de timp adecvata
- **Algoritmul minimax** are rolul de a genera solutia optima pentru castigarea unui joc
- **Algoritmul Alpha Beta** are rolul de a optimiza **minimax-ul**, caci el are o complexitate prea mare in practica, prin renuntarea la anumite drumuri din arborele de cautare (**tehnica de retezare**)
- **Algoritmul minimax**
  - Maparea algoritmului **minimax** la o problema de cautare:
    - stare initiala + cine face prima mutare
    - multimea de operatori posibili
    - inca ceva
  - 2 jucatori: **MIN** (minimizeaza castigul lui MAX) si **MAX** (incearca sa maximizeze castigul)

