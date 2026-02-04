
## Introduction


### State Machine Overview

1. System operational state --> NORMAL
2. Gathering: CHECK, ARE YOU COORDINATOR, COORDINATOR STATE --> NORMAL
3. Merge: START_MERGE, INVITATION, ACCEPT --> ELECTION
4. Setup: REORGANIZATION_MERGE, READY --> REORGANIZATION
5. Termination: FINISH_MERGE --> NORMAL



### During development

- Record demo
- README: Running instructions, principles, flow diagram, TO DO

- read Alexia's documentation & integrate here
- Descriere generala a algoritmului serial
- Descriere algoritm distribuit in format SPMD
- Analiza
  - analiza corectitudine (e.g. care sunt premisele esentiale pentru functionarea algoritmului?)
  - analiza complexitate timp / mesaj
  - avantaje / dezavantaje care apar cu variatia numarului de noduri
  - exemplificare pentru 1-2 topologii la alegere
- Verificare respectare cerinte proiect