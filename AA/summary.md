## Complexities
![Big O](./images/bigO.png)
![Big Omega](./images/bigOmega.png)
![Big Theta](./images/bigTheta.png)
- The following complexity classes work only with **decidable problems**
- $P$ defines the problems solvable in polynomial time by a deterministic Turing Machine
- $NP$ represents the class of problems solvable in polynomial time by a non-deterministic Turing Machine $<=>$ the solution to the problem (the "yes" case) can be found in polynomial time
- $NP-complete$ problems are a special type of $NP$ problems, more abstract, such that any $NP$ problem can be converted into one of them with a polynomial time algorithm
- $NP-hard$ problems are the problems that either are not decidable or the solution cannot be validated in polynomial time
![Complexity classes](./images/complexity_classes.png)
- Pentru problemele pentru care nu avem destule resurse necesare sa le rezolvam optim, avem 2 posibilitati:
  - Algoritmii evolutionist: rulam un algoritm un timp rezonabil si luam cea mai buna solutie
  - Algoritmii $\rho$-aproximativi: rulam programul pana cand solutia este de cel mult $\rho$ ori mai proasta decat cea optima
  - 
