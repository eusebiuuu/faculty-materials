# Instrumente de Comunicare și Sincronizare între Procese (OS)

Sistemele de operare moderne izolează procesele (fiecare are propriul său spațiu de memorie virtuală) pentru a asigura securitatea și stabilitatea sistemului. Totuși, procesele trebuie adesea să colaboreze, motiv pentru care sistemul de operare oferă mecanisme de **Comunicare (IPC)** și, inerent, mecanisme de **Sincronizare**.

---

## Partea I: Concepte de Bază privind Sincronizarea

Înainte de a comunica date complexe, trebuie să ne asigurăm că accesul la resursele partajate este coordonat corect.

* **Secțiunea Critică (Critical Section):** Este porțiunea de cod din interiorul unui proces/thread unde se accesează sau se modifică o resursă partajată (ex. o variabilă globală, un fișier, memorie partajată).
* **Condiția de Cursă (Race Condition):** Este o situație de eroare care apare atunci când mai multe procese sau fire de execuție (threads) accesează și manipulează simultan date partajate, iar rezultatul final depinde de ordinea imprevizibilă în care are loc execuția.
* **Excluderea Mutuală (Mutual Exclusion):** Proprietatea care garantează că, dacă un proces se află în secțiunea sa critică, niciun alt proces nu are voie să intre în secțiunea sa critică corespunzătoare aceleiași resurse.

---

## Partea II: Instrumente de Sincronizare

Sistemul de operare și bibliotecile de programare oferă mai multe primitive pentru a implementa excluderea mutuală și a preveni condițiile de cursă.

### 1. Lacăte (Locks / Mutexes)
* **Mutex (Mutual Exclusion Object):** Funcționează ca o cheie unică pentru o cameră (secțiunea critică). 
* **Regula de aur:** Doar procesul/thread-ul care a "închis" (locked) mutex-ul are dreptul să îl "deschidă" (unlock).
* Dacă un proces încearcă să blocheze un mutex deja blocat, acesta va fi suspendat (pus în așteptare) până când mutex-ul devine disponibil.

### 2. Semafoare (Semaphores)
Inventate de Edsger Dijkstra, semafoarele sunt variabile întregi folosite pentru semnalizare între procese. Ele suportă două operații atomice: `wait()` (sau P / down) și `signal()` (sau V / up).
* **Semafor Binar:** Poate lua valorile 0 sau 1. Este foarte similar cu un mutex, dar diferența fundamentală este că un semafor poate fi incrementat (signal) de *alt* proces/thread decât cel care l-a decrementat (wait).
* **Semafor de Numărare (Counting Semaphore):** Poate lua valori întregi pozitive (ex. $N$). Este folosit pentru a controla accesul la o resursă care are $N$ instanțe identice disponibile (ex. un număr limitat de conexiuni la o bază de date). Când un proces folosește o resursă, scade contorul; când termină, îl crește. Dacă contorul e 0, procesele care cer resursa sunt blocate.

### 3. Variabile de Condiție (Condition Variables)
* Sunt folosite împreună cu un Mutex.
* Ele permit unui proces să elibereze temporar lacătul (mutex-ul) și să "doarmă" (suspendare) până când o anumită *condiție logică* devine adevărată (ex. "coada de mesaje nu mai este goală"). Un alt proces care schimbă starea va trimite un semnal (`signal` sau `broadcast`) pentru a trezi procesul adormit.

### 4. Monitoare (Monitors)
* Este un concept de nivel mai înalt, specific limbajelor de programare (ex. Java cu metodele `synchronized`). 
* Un monitor combină datele partajate, mutex-ul și variabilele de condiție într-un singur construct (un modul sau o clasă), ascunzând detaliile de implementare față de programator. Doar un singur proces poate executa o procedură a monitorului la un moment dat.

---

## Partea III: Instrumente de Comunicare între Procese (IPC)

Acestea sunt metodele efective prin care datele circulă între procesele izolate.

### 1. Conducte (Pipes)
* **Conducte Anonime (Anonymous Pipes):** Permit comunicarea unidirecțională (half-duplex). Datele scrise la un capăt sunt citite la celălalt. Limitare majoră: pot fi folosite DOAR între procese înrudite (ex. un proces Părinte și procesul său Copil creat prin `fork()`).
* **Conducte cu Nume (Named Pipes / FIFOs):** Funcționează similar, dar au un nume în sistemul de fișiere. Aceasta permite proceselor complet independente (neînrudite) să comunice între ele.

### 2. Memoria Partajată (Shared Memory)
* **Principiul:** Sistemul de operare mapează o porțiune din memoria fizică în spațiul de adrese virtuale a două sau mai multor procese.
* **Avantaj:** Este **cea mai rapidă** metodă de IPC, deoarece datele nu trebuie copiate între spațiul kernel-ului (OS) și cel al utilizatorului. Procesele citesc/scriu direct în RAM.
* **Dezavantaj:** Sistemul de operare nu oferă nicio protecție. Programatorul ESTE OBLIGAT să folosească instrumente de sincronizare (Semafoare/Mutex) pentru a preveni rescrierea simultană a datelor.

### 3. Cozi de Mesaje (Message Queues)
* Permite proceselor să trimită "mesaje" formatate (structuri de date) într-o listă menținută de Kernel.
* **Avantaj:** Nu necesită citire imediată (asincron). Mesajele stau în coadă până când procesul destinatar le preia. Nu necesită mecanisme extra de sincronizare, sistemul de operare ocupându-se de concurență.

### 4. Socluri (Sockets)
* Concepute inițial pentru comunicarea prin rețea (între calculatoare diferite folosind IP și porturi).
* **Local Sockets (Unix Domain Sockets):** Sunt folosite pentru IPC pe aceeași mașină fizică. Sunt extrem de fiabile, bidirecționale și suportă atât flux de date (tip TCP) cât și datagrame (tip UDP).

### 5. Semnale (Signals)
* Cel mai primitiv mod de comunicare. Nu transmit date (texte sau numere), ci doar "notificări" asincrone către un proces pentru a-i semnala că un eveniment a avut loc.
* Exemple: `SIGINT` (întrerupere de la tastatură, Ctrl+C), `SIGKILL` (omorârea forțată a procesului), `SIGSEGV` (acces invalid la memorie - Segmentation Fault).


# Planificarea Proceselor (CPU Scheduling) - Cheatsheet

## 1. Concepte de Bază
* **CPU-I/O Burst Cycle:** Execuția oricărui proces constă într-o alternanță între perioade de execuție pe procesor (**CPU burst**) și perioade de așteptare după operații de intrare/ieșire (**I/O burst**).
* **Planificator (Short-term Scheduler):** Componenta sistemului de operare care alege care dintre procesele din starea *Ready* (pregătite de execuție) va primi acces la CPU.
* **Dispecer (Dispatcher):** Modulul care dă efectiv controlul procesorului procesului selectat de planificator. Asta implică:
    * **Context Switch (Schimbare de context):** Salvarea stării procesului vechi (în PCB - Process Control Block) și încărcarea stării procesului nou.
    * Trecerea în modul utilizator (User mode).
    * Saltul la locația corectă din program pentru a relua execuția.
* **Preemptiv vs. Non-preemptiv:**
    * **Non-preemptiv:** Odată ce un proces primește CPU, îl păstrează până când se termină sau trece în starea *Waiting* (așteptând I/O). OS-ul nu îl poate întrerupe forțat.
    * **Preemptiv:** OS-ul poate întrerupe un proces care rulează (de exemplu, când expiră o cuantă de timp sau apare un proces cu prioritate mai mare) și poate da procesorul altui proces.

## 2. Criterii de Performanță (Scheduling Criteria)
Ce urmărim să optimizăm atunci când alegem un algoritm?
1.  **CPU Utilization (Utilizarea procesorului):** Să ținem procesorul ocupat cât mai mult timp posibil (ideal spre 100%).
2.  **Throughput (Debit):** Numărul de procese care își termină execuția într-o anumită unitate de timp (să fie cât mai mare).
3.  **Turnaround Time (Timp de realizare - TT):** Timpul total scurs de la momentul sosirii procesului în sistem până la finalizarea lui completă ($TT = Timp\_Finalizare - Timp\_Sosire$). Trebuie minimizat.
4.  **Waiting Time (Timp de așteptare - WT):** Timpul total petrecut de un proces așteptând în coada *Ready*. Nu include timpul de execuție, doar timpul cât a stat degeaba ($WT = TT - Burst\_Time$). Trebuie minimizat.
5.  **Response Time (Timp de răspuns):** Timpul de la trimiterea unei cereri până la producerea *primului* răspuns (crucial pentru sisteme interactive).

## 3. Algoritmi de Planificare (Scheduling Algorithms)

### A. FCFS (First-Come, First-Served)
* **Tip:** Non-preemptiv.
* **Mecanism:** Procesele sunt planificate exact în ordinea în care sosesc (ca la o coadă la magazin). Implementare cu coadă FIFO.
* **Problemă majoră:** **Efectul de convoi (Convoy Effect)**. Dacă un proces lung ține procesorul, toate procesele scurte din spate vor aștepta mult, rezultând într-un Average Waiting Time foarte mare.

### B. SJF (Shortest-Job-First)
* **Tip:** Poate fi Preemptiv (cunoscut ca **SRTF - Shortest Remaining Time First**) sau Non-preemptiv.
* **Mecanism:** Alege procesul care are cel mai mic *CPU burst* următor. 
* **Avantaj:** Este optim; garantează cel mai mic timp mediu de așteptare (Average Waiting Time).
* **Probleme majore:**
    1.  Este imposibil de știut exact cât va dura următorul burst al unui proces (OS-ul doar estimează bazat pe istoricul procesului).
    2.  **Înfometare (Starvation):** Procesele lungi ar putea să nu primească niciodată procesorul dacă sosesc constant procese scurte.

### C. Round Robin (RR)
* **Tip:** Preemptiv.
* **Mecanism:** Fiecare proces primește o unitate mică de timp pe CPU, numită **cuantă de timp (time quantum - $q$)**, de obicei 10-100 ms. După expirarea cuantei, procesul este scos de pe CPU și pus la coada listei *Ready*.
* **Avantaj:** Foarte bun pentru sisteme time-sharing (interactive), timp de răspuns mic, nu există starvation.
* **Impactul lui $q$:**
    * Dacă $q$ e foarte mare $\rightarrow$ Algoritmul devine FCFS.
    * Dacă $q$ e foarte mic $\rightarrow$ Overhead masiv cauzat de prea multe schimbări de context (Context Switches consumă timp inutil).

### D. Planificarea cu Priorități (Priority Scheduling)
* **Tip:** Preemptiv sau Non-preemptiv.
* **Mecanism:** Fiecărui proces îi este asociată o prioritate. CPU este alocat procesului cu cea mai mare prioritate. (SJF este un caz particular de planificare cu priorități, unde prioritatea = $1/burst$).
* **Problemă majoră:** **Starvation** pentru procesele cu prioritate mică.
* **Soluția:** **Aging (Îmbătrânire)** - creșterea treptată a priorității unui proces pe măsură ce așteaptă în coadă.

### E. Cozi Multinivel (Multilevel Queue & Feedback Queue)
* **Multilevel Queue:** Baza de date a proceselor este împărțită în mai multe cozi distincte (ex. coadă pt. procese sistem, coadă pt. aplicații interactive, coadă pt. batch jobs). Fiecare coadă are propriul ei algoritm de planificare (ex. RR pentru interactive, FCFS pentru batch). Procesele NU se mută între cozi.
* **Multilevel Feedback Queue:** Permite proceselor să migreze între cozi. De exemplu, dacă un proces consumă prea mult CPU time, este mutat într-o coadă cu prioritate mai mică. Dacă un proces blochează pe I/O des, va fi menținut în cozile de prioritate mare. Este cel mai complex și versatil algoritm din sistemele de operare moderne.

# Interblocarea Proceselor (Deadlocks) - Cheatsheet

## 1. Definiție
**Interblocarea (Deadlock)** este o situație în care un set de procese sunt blocate permanent, deoarece fiecare proces din set deține o resursă și așteaptă o altă resursă care este deținută de un alt proces din același set. Niciunul nu poate avansa, rezultând o așteptare circulară infinită.

## 2. Condițiile lui Coffman (Condiții Necesare)
Pentru ca un deadlock să apară, **TOATE cele 4 condiții** de mai jos trebuie să se îndeplinească simultan:
1.  **Excludere Mutuală (Mutual Exclusion):** Cel puțin o resursă este menținută într-un mod nepartajabil (doar un singur proces o poate folosi la un moment dat).
2.  **Deținere și Așteptare (Hold and Wait):** Un proces trebuie să dețină cel puțin o resursă și să aștepte alocarea altor resurse care sunt deținute de alte procese.
3.  **Fără Preempțiune (No Preemption):** Resursele nu pot fi luate cu forța de la un proces. Ele pot fi eliberate doar voluntar de către procesul care le deține, după ce acesta și-a terminat sarcina.
4.  **Așteptare Circulară (Circular Wait):** Există un set de procese în așteptare ${P_0, P_1, ..., P_n}$ astfel încât $P_0$ așteaptă o resursă deținută de $P_1$, $P_1$ o resursă deținută de $P_2$, ..., și $P_n$ așteaptă o resursă deținută de $P_0$.

## 3. Strategii de Tratare a Interblocajelor

### A. Prevenirea Interblocajelor (Deadlock Prevention)
Scopul este de a asigura că **cel puțin una** din cele 4 condiții Coffman NU se poate îndeplini niciodată.
* *Anulare Mutex:* Imposibil pentru resurse hardware (ex. imprimantă).
* *Anulare Hold and Wait:* Procesul trebuie să ceară toate resursele de la început. (Dezavantaj: Utilizare slabă a resurselor, starvation).
* *Anulare No Preemption:* Dacă un proces cere o resursă ce nu e disponibilă, el trebuie să elibereze toate resursele pe care le deține și să reia cererea mai târziu.
* *Anulare Circular Wait:* Se impune o ordonare globală a resurselor. Procesele pot cere resurse doar într-o ordine strict crescătoare a indexului resursei.

### B. Evitarea Interblocajelor (Deadlock Avoidance)
Sistemul cere informații în avans (ex. numărul maxim de resurse de care va avea nevoie un proces). OS-ul verifică dinamic cererile de alocare pentru a se asigura că sistemul nu intră niciodată într-o "stare nesigură".
* **Stare Sigură (Safe State):** Există o secvență de procese (Safe Sequence) în care toate își pot finaliza execuția.
* **Graful de Alocare a Resurselor (Resource-Allocation Graph):** Folosit dacă există o singură instanță din fiecare tip de resursă. O cerere e aprobată doar dacă nu formează un ciclu în graf.
* **Algoritmul Bancherului (Banker's Algorithm):** Inventat de Dijkstra. Folosit pentru instanțe multiple ale aceleiași resurse. Calculează o matrice de nevoi (Need) și verifică dacă resursele disponibile pot satisface procesele rând pe rând.

### C. Detectare și Recuperare (Detection and Recovery)
Sistemul permite apariția deadlock-urilor, dar are un mecanism pentru a le detecta și a se recupera.
* **Detectare:** Folosind *Graful de așteptare (Wait-for graph)* - se caută cicluri (pentru resurse cu o singură instanță).
* **Recuperare - Terminarea proceselor:** Se omoară toate procesele implicate în deadlock sau se omoară rând pe rând până dispare ciclul.
* **Recuperare - Preemptarea resurselor:** Se confiscă o resursă de la un proces (implică *Rollback* - întoarcerea procesului la o stare anterioară sigură).

### D. Ignorarea (Algoritmul Struțului / The Ostrich Algorithm)
* **Abordarea:** Se presupune că deadlock-urile apar foarte rar. OS-ul pur și simplu ignoră problema. Dacă sistemul îngheață, utilizatorul este responsabil să dea restart sau să omoare un proces manual (ex. din Task Manager).
* **Unde se folosește:** În majoritatea sistemelor de operare generale de azi (**Windows, Linux, macOS**), deoarece prevenirea/evitarea este prea scumpă ca timp de procesor pentru utilizarea de zi cu zi.


# Gestiunea Memoriei Proceselor (Memory Management) - Cheatsheet

## 1. Concepte de Bază
* **Adresa Logică (Virtuală):** Adresa generată de CPU în timpul execuției unui program. Procesul "crede" că are memoria doar pentru el, începând de la adresa 0.
* **Adresa Fizică:** Adresa reală din memoria RAM (hardware).
* **MMU (Memory Management Unit):** Componenta hardware care face traducerea (maparea) în timp real din adrese logice în adrese fizice.

## 2. Fragmentarea Memoriei (Memory Fragmentation)
Pe măsură ce procesele sunt încărcate și scoase din memorie, apar "goluri" (spații libere).
* **Fragmentare Externă:** Există suficient spațiu liber total în memorie pentru a satisface o cerere, dar acel spațiu nu este *contiguu* (este spart în bucăți mici).
* **Fragmentare Internă:** Memoria alocată unui proces este puțin mai mare decât a cerut acesta (din cauza alocării în blocuri de dimensiuni fixe). Spațiul rămas liber *în interiorul* blocului alocat este irosit.

## 3. Strategii de Alocare Contiguă (Dinamică)
Când OS-ul trebuie să găsească un gol pentru un proces nou:
* **First-Fit:** Alocă primul gol suficient de mare. (Rapid, dar lasă resturi).
* **Best-Fit:** Caută în toată memoria și alocă cel mai mic gol care este suficient de mare. (Minimizează resturile mari, dar creează fragmentare externă masivă - bucățele foarte mici și inutile).
* **Worst-Fit:** Alocă cel mai mare gol disponibil. (Lasă resturi mari care pot fi folosite ulterior, dar se consumă rapid golurile mari).

## 4. Alocarea Necontiguă (Soluția Modernă)

### A. Paginarea (Paging)
Elimină fragmentarea externă prin împărțirea memoriei în blocuri de dimensiuni egale.
* **Pagini (Pages):** Memoria *logică* este împărțită în blocuri de dimensiune fixă (ex. 4KB).
* **Cadre (Frames):** Memoria *fizică* (RAM) este împărțită în blocuri de aceeași dimensiune ca paginile.
* **Tabela de Pagini (Page Table):** O structură de date menținută de OS pentru fiecare proces. Conține maparea: Pagina Logică $X 
ightarrow$ Cadrul Fizic $Y$.
* **TLB (Translation Lookaside Buffer):** Un cache hardware foarte rapid aflat în interiorul MMU care reține cele mai recente mapări (Pagină $
ightarrow$ Cadru) pentru a accelera procesul (deoarece accesarea Tabelei de Pagini din RAM este lentă).

### B. Segmentarea (Segmentation)
Împarte memoria în funcție de structura logică a programului (cum o vede programatorul), nu în blocuri de mărime fixă.
* **Segmente:** Bucăți de memorie de dimensiuni variabile (ex. Segment de Cod, Segment de Date, Segment de Stivă/Stack).
* Poate suferi de fragmentare externă. Adesea, sistemele moderne combină segmentarea cu paginarea (Segmented Paging).

## 5. Memoria Virtuală (Virtual Memory)
Permite execuția proceselor care nu sunt complet încărcate în memorie. Astfel, memoria logică poate fi mult mai mare decât memoria fizică.
* **Paginare la Cerere (Demand Paging):** Paginile sunt aduse în RAM doar când sunt efectiv accesate de program.
* **Page Fault (Lipsă de Pagină):** Apare când procesorul încearcă să acceseze o pagină care nu este prezentă în RAM (este încă pe HDD/SSD). OS-ul oprește procesul, aduce pagina de pe disc într-un cadru liber din RAM, actualizează tabela și reia instrucțiunea.
* **Zbatere (Thrashing):** O stare critică în care sistemul este suprapopulat cu procese. OS-ul petrece mai mult timp făcând *Swap* (mutând pagini între RAM și disc din cauza Page Fault-urilor frecvente) decât executând instrucțiuni efective. Performanța scade dramatic.

## 6. Algoritmi de Înlocuire a Paginilor (Page Replacement)
Când apare un Page Fault și memoria RAM este plină (nu mai sunt cadre libere), OS-ul trebuie să aleagă o pagină victimă pentru a o da afară.

* **1. FIFO (First-In, First-Out):** Înlocuiește pagina care a fost adusă prima în memorie (cea mai veche).
    * *Dezavantaj:* **Anomalia lui Belady** - pentru anumite șiruri de referință, creșterea numărului de cadre RAM poate duce paradoxal la o *creștere* a numărului de Page Fault-uri!
* **2. OPT / MIN (Optimal):** Înlocuiește pagina care NU va fi folosită pentru cea mai lungă perioadă de timp din viitor.
    * *Performanță:* Ideal, are cel mai mic număr de erori de pagină.
    * *Problemă:* Imposibil de implementat în realitate (OS-ul nu poate prezice viitorul). Este folosit doar ca benchmark teoretic.
* **3. LRU (Least Recently Used):** Aproximație a algoritmului Optimal. Înlocuiește pagina care nu a fost folosită de cel mai mult timp (se uită în trecut, nu în viitor).
    * Este cel mai folosit în practică. Nu suferă de Anomalia lui Belady. Implementarea hardware necesită un contor sau o stivă.
