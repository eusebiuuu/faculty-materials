## Complexities and complexity classes
![Big O](./images/bigO.png)
![Big Omega](./images/bigOmega.png)
![Big Theta](./images/bigTheta.png)
- The following complexity classes work only with **decidable problems**
- $P$ defines the problems solvable in polynomial time by a deterministic Turing Machine
- $NP$ represents the class of problems solvable in polynomial time by a non-deterministic Turing Machine $<=>$ a solution to the problem (the "yes" case) can be verified in polynomial time
- $NP-complete$ problems are a special type of $NP$ problems used to showcase that no solution can be found in polynomial time for a certain problem; any $NP - P$ problem can be converted into a $NP-complete$ problem in polynomial time by a **reduction** algorithm
- $NP-hard$ problems are the problems that either are not decidable or the solution cannot be validated in polynomial time; they include the $NP - complete$ problems because any $NP - P$ problem can be converted into an $NP-hard$ problem through a polynomial time algorithm
- ![Complexity classes](./images/complexity_classes.png)

## Abordarea problemelor grele (NP sau NP-hard)
- Pentru problemele pentru care nu avem destule resurse necesare sa le rezolvam optim, avem 2 posibilitati:
  - Algoritmii evolutionist: rulam un algoritm un timp rezonabil si luam cea mai buna solutie
  - Algoritmii $\rho$-aproximativi: alegem un algoritm care da o solutie ce este de cel mult $\rho$ ori mai proasta decat cea optima
- Astfel, pentru a le rezolva renuntam la una din urmatoarele 3 cerinte:
  - Solutia sa fie optima
  - Solutia sa fie generala
  - Solutia sa fie polinomiala

## Indicatii pentru demonstratia algoritmilor aproximativi
![](./images/aproximative_algorithms.png)
- Define **ALG**
- Asumpti in legatura cu **OPT** in functie de obiectiv (maximizare sau minimizare)
- Ask: When does this algorithm perform poorly?
- Ask: How much better could OPT do?
- Analizarea starii corespunzatoare actiunii curente / ultimei actiuni
- Sortare
- Demonstratii de racapitulat:
  - Knapsack
  - Load balancing
  - Vertex cover
  - TSP
  - Trucks and loads
  - Linear programming

## Algoritmii evolutionisti (genetici)
- Presupun executarea unor programe in mod repetat cu scopul de a obtine o solutie din ce in ce mai apropiata de ce optima
- Initial, dupa ce s-a stabilit intervalul $[a, b]$ se codifica populatia (**populatie** = mulţime de indivizi care trăiesc într-un mediu la care trebuie să se adapteze) initiala cu o precizie $p$ (adica se va imparti in intervale de lungime $\frac{1}{10^p}$, deci vor fi $(b - a)10^p$ intervale) pentru a se afla cromozomii (**cromozom** = mulţime ordonată de elemente (gene) ale căror valoare (alele) determină caracteristicile unui individ) fiecarui individ
- Dupa aceea, se executa programul si se aplica operatorii genetici in functie de probabilitate lor (**incrucisare** si **mutatie**)
- Ulterior, in functie de **functia de fitness** (măsură a gradului de adaptare la mediu pentru fiecare individ) se va selecta (proces prin care sunt promovați indivizii cu grad ridicat de adaptare la mediu) o generatie (etapa in evolutia populatiei) noua
- Selectia poate fi:
  - proportionala (bazata pe fitness)
  - elitista
  - turneu
  - bazata pe ordonare

## Algoritmii probabilisti
- Executia programului se bazeaza pe un set de numere alese aleator
- **Algoritmii Monte Carlo**: ruleaza in timp polinomial si ofera raspuns **probabil** corect
- **Algoritmii Las Vegas**: ruleaz **probabil** in timp polinomial si ofera raspuns corect
- **Algoritmii Atlantic City**: ambele combinate
