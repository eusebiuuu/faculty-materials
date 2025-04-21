## Notiuni generale
- Programul reprezinta un **agent inteligent**
- In rezolvarea unei probleme agentul realizeaza mai multe etape:
  - **Formularea scopului**
    - determinarea setului de **scopuri** (setului de stari destinatie)
  - **Formularea problemei**
    - determinarea **spatiului starilor** si **actiunilor** (nodurilor si muchiilor arborelui corespunzator problemei) care trebuie luate in considerare
    - definirea **starii initiale**
    - **multimea operatorilor**: reprezinta actiunile ce trebuie considerate pentru a ne muta dintr-o stare oarecare
    - **testul scop**: poate fi o conditie de apartenenta, de verificare etc. a starii curente pentru a apartine multimii de stari scop
    - **functia de cost a unui drum**: suma costurilor (individuale) care compun drumul
    - **IMPORTANT**: Pentru problemele cu mai multe stari initiale, starile din acest caz vor forma multimi de stari, iar problema se va adapta corespunzator (intuitie: **DFA -> NFA**)
  - **[Cautarea](#cautarea)**
  - **Executia**: la final, atunci cand au fost obtinute toate solutiile, se va executa actiunea care ajunge in cea optima si se va relua procesul

## Cautarea
- procesul de decizie al agentului se bazeaza pe examinarea starilor in care poate tranzitiona; acest proces se numeste **cautare**
- cautarea se bazeaza pe o **strategie de cautare**, care trebuie sa indeplineasca **completitudinea**, **complexitatea timpului**, **complexitatea spatiului**, **optimalitatea** (Orice algoritm care garantează găsirea unui drum optim la scop este un **algoritm admisibil**.) (ex.: considerarea solutiei cu cea mai mica estimare)
- pentru acest pas se va construi un **arbore de cautare** pentru a determina **drumurile** in **spatiul starilor**
- **IMPORTANT**: arborele de cautare este format din structuri de date numite **noduri**, iar spatiul starilor este alcatuit din **stari** ce reprezinta configuratii ale mediului; de asemenea, 2 noduri pot stoca informatia aceleiasi stari
- Structura de date ce defineste nodurile trebuie sa stocheze urmatoarele informatii:
  - starea referita
  - parintele
  - operatorul folosit pentru a se genera nodul
  - adancimea nodului
  - costul drumului de la starea initiala pana la nod
- In timpul cautarii, exista mai multe noduri care doresc a fi extinse (**frontiera**), ca urmare avem nevoie de o structura de date pentru a realiza selectia celui mai bun nod

## Cautarea best-first
- Fie o functie euristica $f$ ce reprezinta rezultatul evaluarii unei stari, numar real, bazat pe informatii specifice domeniului in care s-a formulat problema
- La fiecare pas se va alege nodul cu functia euristica cea mai mica si se va extinde cu toti succesorii lui
- Totusi, in aceasta maniera nu vom lua in considerare drumurile cu costul mai mic dar cu o functie euristica mai mare
- Ca urmare, $f$ va fi formata din suma dintre $g$ si $h$, unde $g(u)$ reprezinta costul minim de la starea initiala la nodul $u$, iar $h(u)$ este functia euristica de mai sus, adica costul minim al unui drum de la $u$ la un nod scop
- In imagine, se va prezenta un algoritm de cautare care poate fi particularizat cu BF, DF sau best-first (deci si cautare informata si neinformata), numit **GraphSearch**
- **IMPORTANT**: daca spatiul starilor nu este arbore, atunci pot aparea cicluri; ele trebuie detectate in pasul 6
![](./images/GraphSearch.png)

## A*
- Algoritmul A* este o particularizare a acestuia, cu mentiunile ca acea cautare este de tipul best-first, iar nodurile vor fi ordonate in coada in functie de costurile lor
- De asemenea, $f(n) = g(n) + h(n)$, unde $f(n)$ reprezinta costul minim al unui drum de la start la un nod scop, $g(n)$ este drumul de cost minim de la start la $n$, iar $h(n)$ este costul minim efectiv de la $n$ la un nod scop
- Cum nu putem prezice viitorul, definim o functie $h'(n)$ ce are rolul de a oferi o **estimare** pentru $h$, $g'$ ca fiind $g$-ul gasit **pana la pasul curent** din parcurgere si $f'$ distanta minima **pana acum** de la start la un nod scop
- Ca urmare, avem ca $f'(n) = g'(n) + h'(n)$
![](./images/A*algorithm.png)
- A* este un algoritm complet, admisibil si optim.
- Pentru ca A* sa fie admisibil, avem nevoie de 3 conditii:
  - $h'(n) \le h(n)$
  - orice nod are un numar finit de succesori
  - toate costurile sunt pozitive
- Daca $h'(n) \le h(n) \space \forall n$ atunci $h$ este o **euristica admisibila**
- Daca $h(n) \le cost(n, succ) + h(succ)$ atunci $h$ este o **euristica consistenta**

## IDA* (Iterative Deepening A*)
- IDA* reprezinta o optimizare a lui A* care se bazeaza pe monotonia lui $f$.
- Concret, cum $f$ este crescatoare, atunci se pot forma anumite nivele ale nodurilor marginite de costurile lor pana atunci si costul maxim al unui nod succesor unuia de pe nivelul curent. Pentru a forma astfel de **contururi** se va folosi un **DFS**. In acest fel, nu vor mai fi stocate toate nodurile, ci doar cele de pe conturul curent
![](./images/IDA*.png)

## Algoritmii euristici si Teoria Jocurilor
- Incertitudinea din teoria jocurilor, la fel ca in lumea reala, reise din incapacitatea oamenilor sa conceapa solutia optima intr-o perioada de timp adecvata
- Ca urmare, suntem nevoiti sa folosim diverse strategii euristice pentru a elimina cazurile nedorite (**retezare**), astfel incat cautarea unei mutari cat mai bune sa ia cat mai putin timp.
- In urmatoarele situatii, vom prezenta tehnica de cautare prezentata mai sus, aplicata in teoria jocurilor (vom considera jocurile de 2 persoane cu informatie completa), iar apoi ii vom optimiza prin retezare.
### **Algoritmul minimax**
  - 2 jucatori: **MIN** (minimizeaza castigul lui MAX) si **MAX** (incearca sa maximizeze castigul, muta primul)
  - este formata dintr-o stare initiala, o multime de operatori (mutarile), un test terminal si o functie de utilitate (returneaza scorul rezultatului)
  - Fiind extrem de ineficient, se va folosi doar pentru a analiza mutarile pentru cateva nivele (pana cand timpul expira, vezi **Iterative Deepening strategy**)
![](./images/MiniMax.png)
- Algoritmul poate fi optimizat prin 2 moduri:
  - in loc de **functia de utilitate** avem **functia de evaluare** ce reprezinta o estimatie a rezultatului jocului, dintr-o pozitie data
### **Functia de evaluare**
  - aceasta estimatie se bazeaza pe **caracteristicile** unei pozitii
  - o valoare reprezinta o multime de pozitii, ce formeaza o **categorie**
  - $w_1f_1 + w_2f_2 + ... + w_nf_n$, unde $w_i$ reprezinta ponderea pozitiei $i$, iar $f_i$ caracteristicile pozitiei $i$
  - Ponderile pot fi aflate automatizat, prin executarea repetata a jocului dintre program si el insusi, iar mai apoi calibrate dupa fiecare joc
  - In cazul caracteristicilor, ele nu pot fi aflat automatizat
### Alpha Beta Prunning
- Reprezinta o optimizare a algoritmului MiniMax, intrucat integreaza tehnica de retezare
- **Tehnica de retezare** presupune gasirea unei solutii suficient de bune pentru a renunta la o parte din arborele de cautare astfel incat timpul de aflare a deciziei corecte sa fie rezonabil
- Aceasta tehnica este implementata prin existenta variabilelor $\alpha$ si $\beta$ care reprezinta valoarea minima pe care o poate lua MAX, respectiv, cea maxima. Pe parcursul programului, acestea cresc una catre cealalta 
![](./images/alpha-beta.png)

## Reprezentarea cunostintelor (refactoring)
- Cunostinte relationale (ca in bazele de date, se pot baza si pe mostenire)
- Cunostinte inferentiale (logica matematica - rezolutie)
- Cunostinte procedurale
  - Presupun specificarea unor detalii, cum ar fi **ce** anume se intampla si **cand** anume
  - Pentru reprezentarea lor, se folosesc **reguli de productie**, adica structuri de tipul if-then-else, cu diverse interpretari
- **Sistemul expert**
  - Reprezinta un program care se comporta ca un expert intr-o arie restransa, bazandu-se pe cunostintele unor persoane experte reale
  - El este compus dintr-o baza de cunostinte, un motor de inferenta si o interfata grafica.
- Principalele modalitati de reprezentare sunt:
  - **Cele bazate pe logica**: pot fi bazate pe inferenta logica sau bazate pe reguli (mecanisme de inainte de inapoi)
  - **Slot-filler**: se bazeaza pe cautari si reprezentarea lor prin grafuri si pe **pattern matching**

## Retele Bayesiene
- In lumea reala, evenimentele pot fi de 2 tipuri: dependente de alte evenimente sau independente.
- Pentru a putea lucra si modela evenimente dependente vom folosi o **retea Bayesiana**
- Pe scurt, Reteaua Bayesiana constituie un graf directionat aciclic in care fiecare muchie corespunde unei dependente conditionate si fiecare nod corespunde unei variabile aleatoare unice care este reprezentata printr-un tabel de probabilitati conditionate ale parintilor ei.
- Dorim sa construim aceste tipuri de retele cu evenimentele pe care le avem pentru a putea realiza **inferente** (deductii) pe baza lor si probabilitatilor asociate
- Aceste inferente sunt realizate de **algoritmii de inferenta** care se bazeaza pe principiul **independentei coditionate**
- **Independenta conditionata** are loc atunci cand toate drumurile care pleaca dintr-o multime de noduri $X$ sunt blocate **nedirectionata** (adica nu se ia in considerare directia) sa ajunga in $Y$ de o multime de noduri $E$
- Fiind dată o mulţime de noduri E, spunem că un drum este blocat condiţionat de E dacă există un nod Z aparţinând drumului, pentru care una dintre următoarele trei condiţii se verifică:
![](./images/conditional_block.png)
- Aceasta inferenta, pe langa independenta conditionata, se bazeaza si pe probabilitatile conditionate dintre variable
- Concret, fiind data reprezentarea de mai jos a unei retele Bayesiene, dorim sa determinam $P(X | E), E = E^+ \cup E^-$ este definit pe imaginea urmatoare:
![](./images/inference_definition.png)
- 


## Naive Bayes
- Pe baza de Bag of Words
- Fie o variabila aleatoare $Y$ ce reprezinta label-ul si $X_i$ v.a. ce reprezinta feature-urile.
- Astfel, avem formula $P(X_1, ..., X_n | Y) = \prod P(X_i | Y) = 2 ^ {\sum log_2P(X_i | Y)}$