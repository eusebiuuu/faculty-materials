# Reprezentarea cunostintelor

### Notiuni generale
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

### Cautarea
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

### Cautarea best-first
- Fie o functie euristica $f$ ce reprezinta rezultatul evaluarii unei stari, numar real, bazat pe informatii specifice domeniului in care s-a formulat problema
- La fiecare pas se va alege nodul cu functia euristica cea mai mica si se va extinde cu toti succesorii lui
- Totusi, in aceasta maniera nu vom lua in considerare drumurile cu costul mai mic dar cu o functie euristica mai mare
- Ca urmare, $f$ va fi formata din suma dintre $g$ si $h$, unde $g(u)$ reprezinta costul minim de la starea initiala la nodul $u$, iar $h(u)$ este functia euristica de mai sus, adica costul minim al unui drum de la $u$ la un nod scop
- In imagine, se va prezenta un algoritm de cautare care poate fi particularizat cu BF, DF sau best-first (deci si cautare informata si neinformata), numit **GraphSearch**
- **IMPORTANT**: daca spatiul starilor nu este arbore, atunci pot aparea cicluri; ele trebuie detectate in pasul 6
![](./images/GraphSearch.png)

### A*
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

### IDA* (Iterative Deepening A*)
- IDA* reprezinta o optimizare a lui A* care se bazeaza pe monotonia lui $f$.
- Concret, cum $f$ este crescatoare, atunci se pot forma anumite nivele ale nodurilor marginite de costurile lor pana atunci si costul maxim al unui nod succesor unuia de pe nivelul curent. Pentru a forma astfel de **contururi** se va folosi un **DFS**. In acest fel, nu vor mai fi stocate toate nodurile, ci doar cele de pe conturul curent
![](./images/IDA*.png)

### Algoritmii euristici si Teoria Jocurilor
- Incertitudinea din teoria jocurilor, la fel ca in lumea reala, reise din incapacitatea oamenilor sa conceapa solutia optima intr-o perioada de timp adecvata
- Ca urmare, suntem nevoiti sa folosim diverse strategii euristice pentru a elimina cazurile nedorite (**retezare**), astfel incat cautarea unei mutari cat mai bune sa ia cat mai putin timp.
- In urmatoarele situatii, vom prezenta tehnica de cautare prezentata mai sus, aplicata in teoria jocurilor (vom considera jocurile de 2 persoane cu informatie completa), iar apoi ii vom optimiza prin retezare.
#### **Algoritmul minimax**
  - 2 jucatori: **MIN** (minimizeaza castigul lui MAX) si **MAX** (incearca sa maximizeze castigul, muta primul)
  - este formata dintr-o stare initiala, o multime de operatori (mutarile), un test terminal si o functie de utilitate (returneaza scorul rezultatului)
  - Fiind extrem de ineficient, se va folosi doar pentru a analiza mutarile pentru cateva nivele (pana cand timpul expira, vezi **Iterative Deepening strategy**)
![](./images/MiniMax.png)
- Algoritmul poate fi optimizat prin 2 moduri:
  - in loc de **functia de utilitate** avem **functia de evaluare** ce reprezinta o estimatie a rezultatului jocului, dintr-o pozitie data
#### **Functia de evaluare**
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

### Reprezentarea cunostintelor (refactoring)
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

### Retele Bayesiene
- In lumea reala, evenimentele pot fi de 2 tipuri: dependente de alte evenimente sau independente.
- Pentru a putea lucra si modela evenimente dependente vom folosi o **retea Bayesiana**
- Pe scurt, Reteaua Bayesiana constituie un graf directionat aciclic in care fiecare muchie corespunde unei dependente conditionate si fiecare nod corespunde unei variabile aleatoare unice (descrise de un expert) care este reprezentata printr-un tabel (**Conditional probability table / CPT)** de probabilitati conditionate ale parintilor ei.
- **Suportul cauzal** al unei variabile aleatorii (nod în topologie) se referă la evidența sau informația care influențează probabilitatea unei variabile prin intermediul relațiilor cauzale directe sau indirecte reprezentate în rețea. Acesta implică un raționament de sus în jos (de la cauză la efect) sau în lanț prin legăturile cauzale.
- **Suportul probatoriu** al unei variabile aleatorii (nod în topologie) se referă la raționamentul de jos în sus (de la variabilele observate mergând din părinte în părinte spre nodul curent)
- Pe langa acestea, probabilitatile evenimentelor sunt calculate cu relatia lui Bayes, $P(A|B) = \frac{P(B|A) * P(A)}{P(B)}$, unde:
  - P(A|B) este probabilitatea evenimentului A dat fiind că evenimentul B a avut loc (**probabilitatea a posteriori**).
  - P(B|A) este probabilitatea evenimentului B dat fiind că evenimentul A a avut loc (**verosimilitatea**).
  - P(A) este probabilitatea a priori a evenimentului A.
  - P(B) este probabilitatea a priori a evenimentului B (probabilitatea totală).
- Dorim sa construim aceste tipuri de retele cu evenimentele pe care le avem pentru a introduce dovezi (observații despre stările anumitor variabile) și apoi actualiza probabilitățile celorlalte variabile din rețea (fie cauze, fie efecte), calculând probabilitățile a posteriori. Acest proces se numește **inferență Bayesiană** (facem deductii despre alte evenimente).
- Retelele Bayesiene sunt reprezentate ca **poliarbori**. Caracteristica definitorie a unui poliarbore este că între oricare două noduri din graf există cel mult o singură cale (neorientată). De exemplu, stiind ca a nins, dorim sa aflam probabilitatea ca este iarna, avand deja anumite probabilitati cunoscute (nodurile radacina si frunzele).
- Aceste inferente sunt realizate de **algoritmii de inferenta** care se bazeaza pe principiul **independentei conditionate** pentru a optimiza calculele si a reteza subarbori.
- **Independenta conditionata** are loc atunci cand toate drumurile care pleaca dintr-o multime de noduri $X$ sunt blocate **nedirectionat** (adica nu se ia in considerare directia) sa ajunga in $Y$ de o multime de noduri $E$. Altfel spus, două evenimente sau variabile aleatoare A și B sunt independente condiționat pe un al treilea eveniment sau variabilă aleatoare C dacă probabilitatea ca A să se întâmple (sau să ia o anumită valoare), dată fiind informația că B s-a întâmplat (sau a luat o anumită valoare), nu este influențată de faptul că C s-a întâmplat (sau a luat o anumită valoare). Formal, avem ca $P(A \cap B|C) = P(A,B|C) = P(A|C)\cdot P(B|C)$
- În cadrul unei rețele Bayesiene, un drum blocat condiționat (conditionally blocked path) reprezintă o secvență de noduri conectate prin arce (într-o anumită direcție), unde transmiterea influenței probabilistice (sau a dependenței) între două noduri de la capetele drumului este oprită sau blocată de starea cunoscută (condiționată) a unuia sau mai multor noduri intermediare de pe acel drum.
- Fiind dată o mulţime de noduri E, spunem că un drum este blocat condiţionat de E dacă există un nod Z aparţinând drumului, pentru care una dintre următoarele trei condiţii se verifică:  
![](./images/conditional_block.png)
- Exemplificare (**se presupune ca nodurile din Z vor fi observate**):
  - Cazul 1 (X -> Z -> Y sau invers):  
    Daca Z **este observat**, atunci X si Y devin **independente**, altfel sunt **dependente**  
    X = Ploaie  
    Z = Trotuar ud  
    Y = Alunecare.  
    Dacă știm că trotuarul este ud, atunci dacă a plouat sau nu devine irelevant pentru probabilitatea de a aluneca (în măsura în care ploaia influențează alunecarea doar prin faptul că face trotuarul ud). X și Y devin independente condiționate pe Z.
  - Cazul 2 (X -> Z, Y -> Z):
    **Blocare**: Un drum între X și Y este blocat condiționat de E dacă un nod Z sau oricare dintre descendenții lui Z nu sunt observați (**adica nici Z și nici descendenții nu aparțin lui E**).  
    X = Lipsă de somn  
    Y = Stres  
    Z = Durere de cap.  
    Dacă nu știm dacă o persoană are dureri de cap, lipsa de somn și stresul pot fi independente în influența lor asupra altor variabile. Însă, dacă știm că persoana are dureri de cap, atunci a ști că a avut puțin somn ar putea reduce probabilitatea ca stresul să fie cauza principală, introducând o dependență (negativă) între lipsa de somn și stres, condiționată pe durerea de cap.
  - Cazul 3 (Z -> X, Z -> Y):  
    Daca Z **este observat** atunci ele sunt **independente**, altfel ele sunt **dependente**.   
    Z = Defect de fabricație  
    X = Produs defect (tip 1)  
    Y = Produs defect (tip 2).  
    Ambele tipuri de defecte sunt cauzate de același defect de fabricație. Dacă știm că a existat un defect de fabricație, atunci observarea unui produs defect de tip 1 nu ne spune nimic nou despre probabilitatea ca un alt produs să aibă defect de tip 2 (deoarece amândouă sunt deja explicate de defectul de fabricație cunoscut). X și Y devin independente condiționate pe Z.
- **D-separarea** ne spune dacă fluxul de influență probabilistică (sau dependența) între două seturi de variabile este "blocat" de cunoașterea stării unui al treilea set de variabile. Dacă toate căile posibile de influență sunt blocate, atunci cele două seturi devin independente în contextul informației date.
- 2 noduri (sau seturi de noduri) X și Y sunt **d-separate** de un set de noduri E dacă toate căile neorientate între un nod din X și un nod din Y sunt "blocate" de setul E. Două noduri X şi Y sunt independente condiţionat de E, dacă toate drumurile nedirecționate (ignorând sensul arcelor) de la X la Y sunt d-separate de E
- In practica, utilitatea retelelor Bayesiene consta in aflare probabilitatilor unor anumite evenimente conditionate de alte evenimente ale caror probabilitati au fost introduse de experti bazandu-se pe diferite date si experimente. Aceste probabilitati sunt calculate prin diferiti algoritmi de inferenta si folosindu-se anumite formule intre ele. Mai jos se poate gasi modalitatea de notatie a diferitelor componente in functie de evenimentul a carei probabilitate dorim sa o aflam (aici $X$), precum si algoritmul din spatele determinarii ei ($P(X | E), E = E^+ \cup E^-$):
![](./images/inference_definition.png)
![](./images/bayesian_networks_inference_algorithm.png)


# Invatarea automata

### Notiuni generale
- Se foloseste atunci cand nu putem scrie un program pentru determinarea unei anumite functii, deoarece este prea complexa, insa avem output-ul pentru multe input-uri. Ca urmare, vom incerca sa estimam cat mai bine functia respectiva din ce corelatii de input si output avem.
- Pricipalele componente ale unui algoritm de invatare automata sunt:
  - Reprezentarea / Modelarea
  - Evaluarea
  - Optimizarea
- Paradigme de invatare:
  - Supervised learning
  - Unsupervised learning
  - Reinforcement learning
  - Semi-supervised learning
  - Active learning
  - Transfer learning
- Pentru supervised learning avem urmatoarele metode de invatare principale:
  - Clasificatorul Bayes naiv
  - Metoda celor mai apropiați vecini (k-NN)
  - Clasificatorul cu vectori suport
  - Metode kernel
  - Regresii: Liniara, Ridge, Lasso
  - Rețele neuronale și învățare "deep"
  - Arbori de decizie și random forests
- Pentru supervised learning avem urmatoarele forme canonice:
  - Regression
  - Classification
![](./images/supervised_learning.png)
- **Unsupervised learning** works with **unlabeled** data: the algorithm tries to find hidden patterns or structures without any prior knowledge of correct outputs.
- Principalele metode de unsupervised learning sunt:
  - K-means clustering
  - **PCA**
  - Clustering ierarhic
  - Analiza în componente principale
  - Modele de tip auto-encoder
  - Anomaly detection
  - Association rule learning
- Pentru unsupervised learning avem urmatoarele forme canonice:
  - Clustering
  - Dimension reduction
- **Invatarea activa** presupune alegerea unui subset mic dintr-un set de date neetichetate pe care să îl etichetăm pentru a obține un clasificator cât mai bun.
- **Transfer learning** presupune reutilizarea unui model antrenat pe un anumit domeniu / problema pe un alt domeniu / problema

To understand machine learning, you need to be familiar with its foundational vocabulary:
* **Dataset (Set de date):** The collection of data used to train and evaluate the model. It is typically divided into a **Training Set** (used to teach the model) and a **Testing Set** (used to evaluate its performance).
* **Features / Attributes (Caracteristici):** The individual, measurable properties or variables of the phenomena being observed. (e.g., in a dataset about cars, features might be mileage, age, and brand).
* **Target / Label (Etichetă):** The output variable you are trying to predict (e.g., the price of the car).
* **Model (Model):** The mathematical representation or mathematical function learned by the algorithm that maps inputs (features) to outputs (predictions).
* **Training / Learning (Antrenare):** The process where the algorithm analyzes the training data to find patterns and adjust its internal parameters to minimize errors.
* **Inference / Prediction (Predicție):** Applying the trained model to new, unseen data to guess the output.

### Canonical Forms of Learning Problems (Forme canonice ale problemelor de învățare)
Depending on the type of data and the desired outcome, machine learning problems usually fall into one of these standard forms:

#### Under Supervised Learning:
* **Classification (Clasificare):** The task of predicting a **discrete class label** or category. 
    * *Example:* Identifying if an email is "Spam" or "Not Spam"; diagnosing a tumor as "Malignant" or "Benign".
* **Regression (Regresie):** The task of predicting a **continuous numerical value**.
    * *Example:* Predicting the future price of a house based on its size and location; forecasting tomorrow's temperature.

#### Under Unsupervised Learning:
* **Clustering (Grupare):** The task of grouping a set of objects in such a way that objects in the same group (cluster) are more similar to each other than to those in other groups.
    * *Example:* Customer segmentation based on purchasing behavior.

### Structura invatarii programelor
- Procesul de invatare al programelor se realizeaza in 3 etape: antrenarea, validarea si testarea. Fiecare va avea un set specific alocat din dataset-ul furnizat
- Validarea are rolul de a produce o estimare mai buna a hiperparametrilor programului. Daca am face testarea hiperparametrilor pe set-ul de test, atunci am putea ajunge la overfitting
![](./images/learning_stages.png)
- O modalitate similara este cea de **cross-validation** care presupune divizarea setului de date de antrenament in $k$ parti, fiecare parte, prin rotatie, ajungand sa fie multimea de validare
![](./images/cross_validation.png)

### Evaluarea performantei
- **Accuracy**: Overall correctness. Not very good usually.
- **Precision**: How many of the predicted positives are actually positive: tp / (tp + fp)
- **Recall**: How many of the actual positives were correctly predicted: tp / (tp + fn)
- **F1 Score**: Harmonic mean of precision and recall: $\frac{(1 + \beta^2)\cdot precision \cdot recall}{\beta^2 \cdot precision + recall}$
- **Confusion Matrix**: Shows TP, FP, TN, FN for deeper analysis.
- **Jaccard Index**: Overlap between predicted and actual labels.
- **ROC-AUC**: For binary classifiers — tradeoff between true and false positives.
- **MSE / MAE**: For regression. Depends on how much we want to punish the mistakes.
- **Kendall’s Tau**: Is a non-parametric measure of the strength and direction of association between two rankings.
- **Model Parameters**: These are internal to the model and are learned from the training data. In SVMs, w and b define the decision boundary that separates different classes.
- **Hyperparameters**: These are external configurations set before the training process begins. They control aspects of the training process and model complexity.
- **Mean Absolute Error (MAE):** The average of the absolute differences between predictions and actual values.
- **Mean Squared Error (MSE):** The average of the *squared* differences between predictions and actual values. It penalizes large errors more heavily than MAE.
- **R-squared (R²):** Represents the proportion of variance in the dependent variable that is predictable from the independent variables. Closer to 1.0 is better.


### Clasificatorii
- Clasificatorii sunt programe care, antrenandu-se pe anumite dataset-uri ce au label-uri, incearca sa invete cum sa clasifice cat mai bine datele pe label-uri astfel incat sa fie cat mai aproape de realitate.

### Clasificatorul Naive Bayes
- **Clasificatorul Bayes** realizeaza clasificarea datelor in functie de probabilitatile cu care aceastea (sau parti din ele, numite **feature-uri**) au loc in cadrul dataset-ului de antrenament (de ex, vezi exercitiul cu imagini si pixeli). Altfel spus, analizand datele, el returneaza clasa $C$ cea mai probabila stiind ca avem ca test set-ul de date $D$, adica ne da clasa $C$ pentru care $P(C | D)$ este maximizata. Acest lucru este realizat prin formula lui Bayes si utilizand asumptia **feature-urile** lui $D$ sunt conditional independente, astfel incat formula sa fie mai usor de calculat (de aceea ii zice naiv, pentru ca de cele mai multe ori in realitate nu sunt conditional independente).
- Un alt aspect important al clasificatorului Bayes reprezinta calcularea efectiva a probabilitatilor. Pentru acest lucru ar trebui sa stim cat mai bine distributia feature-urilor unui set de date. Cele mai comune distributii ale feature-urilor, impreuna cu utilizarile lor sunt urmatoarele:
  - **GaussianNB**: utilizat in cazurile in care feature-urile sunt continue; $P(x_i | Y) = \frac{1}{\sqrt{2\pi \sigma_Y^2}} \exp\left(-\frac{(x_i - \mu_Y)^2}{2\sigma_Y^2}\right)$
  - **MultinomialNB**: atunci cand lucram cu feature-uri discrete (ex.: inaltimi ale unor oameni in cm); $P(x_i | Y) = \frac{\text{count}(x_i, Y) + \alpha}{\sum_{j=1}^n \text{count}(x_j, Y) + \alpha \cdot n}$
  - **BernoulliNB**: pentru situatiile cand avem feature-urile au valori binare (ex.: daca o persoana traieste sau nu); $P(x_i | Y) = \begin{cases} p_{iY} & \text{if } x_i = 1, \\ 1 - p_{iY} & \text{if } x_i = 0, \end{cases}$
  - **CategoricalNB**: folosit la fel ca **MultinomialNB** in cazurile in care in loc de numere sunt denumiri de categorii; $P(x_i | Y) = \frac{\text{count}(x_i, Y) + \alpha}{|Y| + \alpha \cdot n}$
  - **ComplementNB**: folosit atunci cand setul de date este biased catre o clasa, el avand astfel rolul de a echilibra intr-o anumita masura probabilitatile finale (eg.: predictia unor boli foarte rare); este folosit doar daca datele de intrare sunt discrete; $w_{yi} = \log \frac{ \sum_{d \notin y} \text{count}(x_i, d) + \alpha }{ \sum_{d \notin y} \sum_{j=1}^n \text{count}(x_j, d) + \alpha \cdot n }$
- In order to prevent unseen features in dataset we use a **smoothing parameter**, usually marked as $\alpha$

### K-Nearest Neighbours (KNN)
- Acest clasificator se foloseste de feature-urile si label-urile din training dataset pentru a calcula cele mai apropiate k feature-uri de feature-ul de test, folosindu-se de o anumita distanta (de obicei l1 sau l2)
- Dintre feature-urile obtinute (cele mai apropiate) se calculeaza label-ul majoritar / cel mai frecvent care va deveni label-ul feature-ului de test
- Aceasta metoda este **dataset sensitive**, deoarece, de la un moment dat, cu cat crestem numarul dimensiunilor, cu atat cresc inacuratetea si complexitatea de timp a algoritmului. In plus, daca setul de antrenare are o distributie diferita fata de cel de testare, vom avea si aici probleme cu acuratetea

### Blestemul dimensionalitatii
- Adăugarea mai multor caracteristici păstrând dimensiunea setului de antrenare degradează performanța clasificatorului
- Atunci când distanța dintre observații crește, învățarea automată devine mult mai dificilă, deoarece scade probabilitatea de a găsi exemple de antrenare cu adevărat similare cu cele de test

### SVM
- **SVM** este un algoritm de **supervised learning**, specific pentru clasificare si regresie.
- It finds the best hyperplane that separates data into classes. The “best” hyperplane is the one with the maximum margin — the greatest distance between the hyperplane and the nearest data points (called support vectors).
- Concret, fiind dati o multime de vectori de antrenare $x_i$ si label-uri $y_i$ pentru fiecare in parte, dorim sa gasim un hiperplan de forma $wx + b = 0$, unde $w$ este normala hiperplanului (adica perpendiculara pe hiperplan) si $b$ este bias-ul. Pentru ca datele sa fie corect clasificate mai avem nevoie si de conditia $y_i(wx_i + b) \ge 1 - \xi_i$. Daca $\xi_i = 0$ atunci avem **hard margin**, deoarece nu se permite nicio clasificare gresita, oricat de mica ar fi (de obicei duce la **overfitting**). In practica aceste erori de clasificare sunt controlate prin parametrul **C**.
- De asemenea, cum dorim sa maximizam marginea, trebuie sa minimizam valoarea $\frac{1}{2} ||w||^2$.

### The Kernel Method & The "Kernel Trick"
Often, the data provided is **not linearly separable** in its original space (e.g., the XOR problem, or concentric circles). In standard linear models, this would cause the algorithm to fail. To solve this, we use the **Kernel Method**.

* **The Core Idea (Scufundarea):** We map the original data from its input space ($x \in \mathbb{R}^m$) into a much higher-dimensional feature space ($F$) using a mapping function $\phi(x)$. In this new higher-dimensional space, the data becomes linearly separable by a hyperplane.
* **The Computational Problem:** If the new space $F$ has millions of dimensions (or is infinite), calculating the dot product $\phi(x)^T \phi(y)$ becomes computationally impossible.
* **The Kernel Trick (Trucul Kernel):** A **Kernel Function** $K(x, y)$ calculates the dot product (similarity) of two points in the high-dimensional space *without* ever explicitly computing the mapping $\phi$. 
    * *Formal Definition:* A function $K: X \times X \rightarrow \mathbb{R}$ is a valid kernel if there exists a mapping $\phi: \mathbb{R}^m \rightarrow F$ such that:
        $K(x, y) = \langle \phi(x), \phi(y) \rangle$
* **Mercer's Theorem (Teorema lui Mercer):** Not every mathematical function can be a kernel. A function qualifies as a valid kernel only if it is symmetric $K(x,y) = K(y,x)$ and produces a positive semi-definite **Gram Matrix** (the matrix containing the kernel evaluations for all pairs of training examples).

### The Decision Function & Support Vectors
The decision function dictates how a new test point is classified. Your notes mentioned a sequence $\alpha$ of length $n$. These $\alpha_i$ values are the **Lagrange Multipliers** learned during training.

The correct, standard formula for the decision boundary (often used in Support Vector Machines) incorporates the true labels ($y_i$) of the training data:
$$f(x^{\text{test}}) = \text{sign} \left( \sum_{i=1}^n \alpha_i y_i K(x_i, x^{\text{test}}) + b \right)$$

* **Support Vectors:** In practice, most $\alpha_i$ values become $0$. The training examples $x_i$ where $\alpha_i > 0$ are called **Support Vectors**. These are the *only* data points that matter for defining the decision boundary. If you delete all other data points, the model remains exactly the same.

### Common Kernel Functions
Different kernels map data into different types of feature spaces.

* **Linear Kernel:** $K(x, x') = x^T \cdot x'$
    * *Explanation:* Does not map to a higher dimension ($\phi(x) = x$). Used when data is already linearly separable or when there are a massive number of features (e.g., text classification).
* **Polynomial Kernel:** $K(x, x') = (\gamma x^T \cdot x' + c)^d$
    * *Parameters:* $d$ is the degree of the polynomial, $c$ is a constant trading off the influence of higher-order versus lower-order terms.
    * *Usage:* Commonly used in image processing.
* **RBF (Radial Basis Function) / Gaussian Kernel:** $K(x,x') = \exp(-\gamma \|x - x'\|^2)$
    * *Explanation:* This maps the data into an **infinite-dimensional space**. It measures the Euclidean distance between points.
    * *The $\gamma$ (gamma) parameter:* Defines the "spread" of the kernel. 
        * **High $\gamma$:** The similarity radius is narrow. Only points very close to each other are considered similar. Leads to highly irregular, complex decision boundaries (**Risk of Overfitting**).
        * **Low $\gamma$:** The similarity radius is wide. Many points influence the decision. Leads to a smoother, more linear decision boundary (**Risk of Underfitting**).

### Primal vs. Dual Forms in Optimization
In machine learning (especially Support Vector Machines), the math used to find the best model can be formulated in two equivalent ways.

#### The Primal Problem (Forma Primală)
* **Perspective:** The "direct" approach. You are trying to find the physical attributes of the decision boundary itself.
* **Goal:** Optimize the main variables. For an SVM, this means finding the optimal weight vector $w$ and bias $b$ that define the hyperplane, while minimizing classification errors.
* **Formula (Soft-Margin SVM):** $$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i$$
    *(Subject to the constraint that data points are classified correctly, where $\xi_i$ are slack variables for errors).*
* **Complexity:** The computational cost depends on $m$, the **number of features (dimensions)**.

#### The Dual Problem (Forma Duală)
* **Perspective:** The "indirect" approach. Instead of finding the hyperplane's weights directly, we look at the interactions (similarities) between the data points themselves.
* **How it works:** We introduce **Lagrange Multipliers ($\alpha_i$)** for every constraint in the primal problem. These act as penalties. The goal switches to maximizing a function based entirely on these multipliers.
* **Formula:** $$\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$
    *(Subject to $0 \le \alpha_i \le C$ and $\sum \alpha_i y_i = 0$)*
* **Complexity:** The computational cost depends on $n$, the **number of data points (examples)**, NOT the number of features.

#### Why do we care about the Dual Form? (Crucial for Exams)
1.  **It enables the Kernel Trick:** Look at the dual formula. The data points $x_i$ and $x_j$ *only* appear as a dot product inside the kernel function $K(x_i, x_j)$. The Primal form requires $w$, which would be infinite-dimensional if we used an RBF kernel. The Dual form bypasses this entirely!
2.  **Dimensionality Advantage:** If you have data with millions of features but only 1,000 samples (like in genetics), the Primal form is mathematically brutal. The Dual form is much faster because it only scales with the 1,000 samples.

### Classification types
- Algoritmii prezentati pana acum merg pe abordarea **One vs One**. Aceasta presupune realizarea separarii dintre toate perechile de 2 clase stabilite.
- Pe langa aceasta exista si metoda **One vs All** ce presupune separarea unei clase de toate celelalte in mod direct. Pentru asta se foloseste **analiza liniar discriminanta** ce presupune aproximarea unei distributii Gaussiene pentru fiecare clasa si apoi definirea unui hiperplan pe care sa se proiecteze punctele a.i. 1) distanta dintre clase sa fie maximizata si 2) dispersia unei clase sa fie minimizata

### Regresii
- In multe situatii suntem nevoiti sa determinam o functie care modeleaza (interpoleaza) cel mai bine datele pe care le avem. Pentru aceasta folosim metode de regresie (liniara, Ridge, Lasso) care presupun estimarea functiei minimizand penalizarile abaterilor de la functia obtinuta.
- Concret, dorim sa determinam o functie $y = w^Tx + b$ unde $w$ este vectorul coeficient (panta) format din $(w_1, w_2 ...)$ pe care modelul doreste sa-l calculeze, $x = (x_1, x_2, ...)$ este vectorul dat, iar $b$ este bias-ul / intercept-ul. De asemenea, vrem sa o facem in asa fel incat sa minimizam **MAE** si **MSE**
- Totusi, in regresia liniara, adica cea prezentata mai sus, in cazul in care avem zgomot sau feature-urile au valori mari, modelul nostru poate face overfitting. Pentru aceasta, au fost create regresiile **Ridge** si **Lasso**, care presupun o regularizare a datelor prin introducerea unor penalizari pentru valorile $w_i$ prea mari calculate. In regresia Ridge, the larger the $\alpha$, the more we shrink the coefficients toward 0 (but not exactly 0). Regresia Ridge presupune o penalizare bazata pe `L2`, in timp ce regresia Lasso se bazeaza pe `L1`.

### Rețele Neuronale Feedforward (FNNs)

#### Perceptronul și Baza Matematică
Rețelele neuronale moderne se bazează pe conceptul de **perceptron** (mai exact, perceptronul multistrat). La nivelul unui singur neuron artificial, operația este similară cu un clasificator liniar sau o regresie liniară. 
Calculul de bază al unui neuron (înainte de activare) se notează adesea cu $z$:
$$z = w^T x + b = \sum_{i=1}^{n} w_i x_i + b$$
* $x$ = vectorul de input (sau output-ul stratului anterior).
* $w$ = vectorul de ponderi (weights) corespunzător conexiunilor.
* $b$ = bias-ul (termenul liber). Acesta translatează funcția de activare spre stânga sau dreapta, ajutând modelul să se potrivească mai bine pe date (acționează ca un prag de declanșare).

#### Funcții de Transfer (Activare) și Derivatele lor
Dacă am folosi doar operația de mai sus, indiferent de câte straturi am adăuga, rețeaua ar fi doar o mare funcție liniară. Pentru a modela relații complexe (neliniare), aplicăm o **funcție de activare (sau de transfer)** peste rezultatul $z$: $a = f(z)$.
Derivata funcției de activare, $f'(z)$, este crucială în procesul de învățare (Backpropagation) pentru calcularea gradientului.

* **Identity (Liniară):** $f(z) = z$
    * *Derivata:* $f'(z) = 1$
    * *Utilizare:* În stratul de output pentru probleme de regresie continuă. Nu se folosește în hidden layers.
* **Sign (Treaptă / Heaviside):** $f(z) = 1$ dacă $z \ge 0$, altfel $-1$ (sau $0$)
    * *Derivata:* $f'(z) = 0$ aproape peste tot.
    * *Utilizare:* Istorică, pentru perceptronul simplu binar. Ineficientă azi deoarece derivata 0 oprește învățarea (gradientul dispare).
* **Sigmoid (Logistică):** $f(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$
    * *Derivata:* $f'(z) = f(z) \cdot (1 - f(z))$
    * *Utilizare:* Clasificare binară (în output layer), transformă output-ul în probabilități $(0, 1)$. 
    * *Dezavantaj:* Suferă de problema "vanishing gradient" (gradientul devine foarte mic pentru valori extreme ale lui $z$).
* **Tanh (Tangenta Hiperbolică):** $f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
    * *Derivata:* $f'(z) = 1 - \tanh^2(z)$
    * *Utilizare:* Bună pentru hidden layers. Este preferată funcției Sigmoid deoarece este centrată în 0 (rezultate între $-1$ și $1$), ajutând la o convergență mai rapidă.
* **ReLU (Rectified Linear Unit):** $f(z) = \max(0, z)$
    * *Derivata:* $f'(z) = 1$ pentru $z > 0$; $f'(z) = 0$ pentru $z \le 0$
    * *Utilizare:* Standardul de facto pentru **hidden layers**. Este rapidă computațional și rezolvă în mare parte problema vanishing gradient.
    * *Dezavantaj:* "Dying ReLU" - neuronii pot deveni inactivi permanent dacă primesc doar valori negative.
* **Leaky ReLU:** $f(z) = z$ (pt $z > 0$), $\alpha z$ (pt $z \le 0$), unde $\alpha$ e o constantă mică (ex. 0.01).
    * *Derivata:* $1$ (pt $z > 0$) și $\alpha$ (pt $z \le 0$). Rezolvă problema "Dying ReLU".
* **Softmax:** $f(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$
    * *Utilizare:* Stratul de output pentru **clasificare multi-clasă**. Transformă un vector de numere într-o distribuție de probabilități a căror sumă este 1.

#### Arhitecturi: Rețele Feedforward (FNNs)
O rețea Feedforward (sau Multi-Layer Perceptron - MLP) direcționează informația într-un singur sens (de la input la output), fără bucle.
* **Input Layer:** Nu face procesare, doar preia setul de feature-uri ($x_1, x_2, ...$). Numărul de neuroni = numărul de feature-uri.
* **Hidden Layers (Straturi ascunse):** Straturile intermediare. O rețea "Deep" are 2 sau mai multe astfel de straturi. Acestea extrag trăsături și modele complexe din date. Matematic, reprezintă o matrice de ponderi $W^{(l)}$ și un vector de bias-uri $b^{(l)}$ pentru fiecare strat $l$.
* **Output Layer:** Formulat în funcție de problemă (1 neuron liniar pt regresie, 1 neuron Sigmoid pt clasificare binară, $N$ neuroni Softmax pt $N$ clase).

#### Inițializarea Datelor (Ponderilor)
Cum alegem valorile inițiale pentru $W$ și $b$ înainte de a începe antrenarea?
* **NICIODATĂ TOATE ZERO:** Dacă inițializăm toate ponderile cu 0 (sau cu aceeași valoare), toți neuronii din stratul ascuns vor învăța aceleași trăsături și vor avea aceiași gradienți (Problema Simetriei).
* **Random Initialization:** Ponderile primesc valori aleatoare mici. Bias-urile pot fi inițializate cu 0.
* **Xavier / Glorot Initialization:** Folosită optim alături de activări **Tanh sau Sigmoid**. Extrage valori dintr-o distribuție calculată pe baza numărului de neuroni de intrare și ieșire.
* **He Initialization:** Folosită optim alături de activări **ReLU / Leaky ReLU**. Previne dispariția semnalului în rețele adânci.

#### Funcții de Cost și Pierdere (Loss Functions)
Procesul de învățare necesită măsurarea erorii rețelei. **Loss** reprezintă eroarea pe un singur exemplu, în timp ce **Cost** este media erorilor pe întregul dataset de training.
* **Regresie:** MSE (Mean Squared Error), MAE (Mean Absolute Error), Huber Loss.
* **Clasificare Binară:** Binary Cross-Entropy (BCE).
* **Clasificare Multi-Clasă:** Categorical Cross-Entropy (CCE).
* **Altele:** Focal Loss (pt date dezechilibrate), Contrastive Loss (Metric learning), KL Divergence (pentru compararea distribuțiilor de probabilitate).

#### Optimizarea: Gradient Descent și Variante
Pentru a minimiza funcția de Cost ($J(W, b)$), trebuie să actualizăm ponderile în direcția inversă a gradientului. 
* Formula de actualizare a unei ponderi: $w_{nou} = w_{vechi} - \eta \cdot \frac{\partial J}{\partial w}$, unde $\eta$ este rata de învățare (Learning Rate).
* **Batch Gradient Descent:** Calculează gradientul pe TOATE datele înainte de un update. Foarte lent și costisitor.
* **Stochastic Gradient Descent (SGD):** Actualizează ponderile după calcularea gradientului pentru **un singur exemplu** (ales aleator). Foarte rapid, dar zgomotos.
* **Mini-Batch SGD:** Compromisul perfect. Calculează gradientul pe un grup mic de exemple (ex. 32, 64, 128). Este standardul modern.

#### Backpropagation și Regula de Înlănțuire (Chain Rule)
**Backpropagation (Propagarea înapoi a erorii)** este algoritmul care calculează eficient gradienții (derivata parțială a funcției de cost în raport cu fiecare pondere și bias din rețea).
El se bazează fundamental pe **Regula de Înlănțuire (Chain Rule)** din analiza matematică.

Dacă avem un neuron de output unde input-ul a fost $z = wx+b$, activarea $a = f(z)$, iar pierderea calculată e $L(a, y)$, vrem să aflăm cât de mult influențează ponderea $w$ pierderea $L$, adică $\frac{\partial L}{\partial w}$.
Aplicând Chain Rule:
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$
Unde:
1.  $\frac{\partial L}{\partial a}$: Derivata funcției de loss (cât de greșit e rezultatul final).
2.  $\frac{\partial a}{\partial z} = f'(z)$: Derivata funcției de activare (de aici importanța ecuațiilor de la punctul 2).
3.  $\frac{\partial z}{\partial w} = x$: Input-ul primit de neuron.

Această eroare ("delta") se calculează mai întâi la Output Layer și se **propagă înapoi**, strat cu strat, spre Input Layer, reutilizând calculele deja făcute pentru a eficientiza algoritmul.

#### Algoritmul Widrow-Hoff (Delta Rule)
Este predecesorul istoric al Backpropagation. Se aplica rețelelor cu un singur strat (Adaline) și cu funcție de activare identitate (liniară). 
Algoritmul presupune actualizarea pe loc a ponderilor proporțional cu eroarea reziduală:
$$\Delta w_i = \eta (y_{target} - y_{predict}) x_i$$
Astăzi este considerat un caz particular, simplificat, al algoritmului general de gradient descent.


### **Convolutional neural networks (CNNs) (OPTIONAL)**
- Retelele convolutionale se bazeaza pe operatia de **convolutie** dintre o submatrice a input-ului (**care va fi mereu o matrice**) si o matrice numita **kernel**, ce reprezinta de fapt inmultirea element cu element a celor 2. Output-ul acestei operatii va fi o matrice de dimensiunea kernel-ului si va reprezenta un **feature map**.
- Un input poate fi format din mai multe dimensiuni, adica mai multe astfel de matrici, in care fiecare dimensiune este responsabila cu un tip de feature (de ex. imaginile RGB), caz in care se vor defini mai multe **canale**, unul pentru fiecare matrice, fiecare cu propriul kernel (sau nu), ce la final se vor uni si vor forma **output-ul layer-ului**.
- Inmultirea din operatia de convolutie se realizeaza prin sliding-ul matricii kernel de-a lungul input-ului. Pasul cu care se realizeaza acest sliding se numeste **stride**.
- For example, given an input of $n\times n$ size, the new size will be $\frac{n - k + 2 \cdot p}{stride} + 1$
- Totusi, se poate observa ca valorile aflate pe marginile matricei nu sunt luate in calcul prea des. Atunci cand ele sunt importante, insa, asta devine o problema. Pentru a o rezolva vom adauga un padding pentru input si, pe noua matrice, vom aplica identic operatia de convolutie.
- Aceste notiuni prezentate alcatuiesc **convolutional layer**-ul ce are rolul de a extrage feature-uri.
- De multe ori insa nu avem nevoie de toata imaginea, ci doar de parti din ea, in principal pentru a mari eficienta algoritmului. Pentru aceasta vom crea **pooling layer-ul** care are rolul de a reduce din dimensiunile fiecarui canal. Procesul de realizare este unul similar celui de la convolutie, doar ca nu se vor mai aduna numere, ci doar se va alege unul (de obicei cu functia de **MAX** sau **AVG**).
- Dupa ce am terminat de procesat canalele input-ului dorim sa le unim. Pentru asta vom aplica **flattening layers** ce au rolul de a transforma matricile in vector 1D pentru ca **Dense layer**-ele sa poata lucra cu datele obtinute
- **Dense layer**-ele au rolul de a analiza datele primite pentru detectarea de feature-uri si integrarea lor in predictii
![](./images/CNN.png)
![](./images/convolutional_block.png)
- In ceea ce priveste functia de activare, ea este prezenta doar la **convolutional layers** si **dense layers**
- **Hiperparametrii**: stride (S), kernel size (K), padding (P)

### Other types of neural networks
- **Long short-term memory networks**: for voice recognition
- **Recurrent Neural Networks (RNNs)**