
## State Machine Replication (SMR)

- State machine replication refers to the transformation of an algorithm from running in an isolated instance to being executed in a distributed environment by replicating the program to each instance from the system.
- This transformation is needed in order to have a fault-tolerant system (that is, when a certain set of instances doesn't work anymore, the system still can work properly, requests being redirected to other working nodes)
- One challenge with this approach is coordinating how the machine communicate (usually, how the client request is handle within multiple servers)
- A **State Machine** is defined by the following tuple:
  - A set of States
  - A set of Inputs
  - A set of Outputs
  - A transition function (Input × State → State)
  - An output function (Input × State → Output)
  - A distinguished State called Start.
- For a **Fault Tolerant** system composed of state machines to accept up to $F$ failures, it has to have at least $2F + 1$ states; this way, when $F$ fail and throw an unpredicted output, there are $F + 1$ left which are sufficient for the system to figure out if an instance failed or not (because there is a majority and the detecting mechanism is based on comparisons)
- The SMR approach must respect the following rules:
  1. Place copies of the State Machine on multiple, independent servers.
  2. Receive client requests, interpreted as Inputs to the State Machine.
  3. Choose an ordering for the Inputs.
  4. Execute Inputs in the chosen order on each server.
  5. Respond to clients with the Output from the State Machine.
  6. Monitor replicas for differences in State or Output.
- The crucial part of implementing such a system resides in the step 3, that has the role of synchronizing the order of Input registry across all the entities such that each node can execute the same set of instructions
- Since all non-faulty replicas will arrive at the same State and Output if given the same Inputs, it is imperative that the Inputs are submitted in an equivalent order at each replica
- As many ordering rules exist (visible channels, hidden channels), one of the best method for open systems represents **Consensus ordering**
- A **Consensus** represents the procedure, between the system entities, of choosing a single value through **majority voting**
- In this way, inputs may be ordered by their position in the series of consensus instances (**Consensus Order**). Moreover, this Consensus Order can be derived independently by each individual machine, since the Input instructions will be executed in exactly the same manner on each instance


## Paxos Introduction

- Paxos is a family of consensus protocols used by SMR systems in order to assure a syncronization of Inputs between unreliable and fallible machines
- The Paxos protocol was first submitted in 1989 and named after a fictional legislative consensus system used on the Paxos island in Greece, where Lamport wrote that the parliament had to function "even though legislators continually wandered in and out of the parliamentary Chamber"
- Although no deterministic fault-tolerant consensus protocol can guarantee progress in an asynchronous network (a result proved in a paper by Fischer, Lynch and Paterson), Paxos guarantees safety (consistency), and the conditions that could prevent it from making progress are difficult to provoke.
- Characteristics (first 2 are guaranteed):
  - Validity (or non-triviality): Only proposed values can be chosen and learned
  - Agreement (or consistency, or safety): No two distinct learners can learn different values (or there can't be more than one decided value).
  - Termination (or liveness): If value C has been proposed, then eventually learner L will learn some value (if sufficient processors remain non-faulty).
- Note that Paxos is not guaranteed to terminate, and thus does not have the **liveness** property. This is supported by the Fischer Lynch Paterson impossibility result (FLP) which states that a consistency protocol can only have two of **safety**, **liveness**, and **fault tolerance**; thus, because Paxos must assure **fault-tolerance** and promise to be **consistent**, the **liveness** cannot be guaranteed


## Basic Paxos (theoretical)

- A typical deployment makes sure the nodes act in three roles: **Learner**, **Acceptor** and **Proposer**; they can be even multiple at once since the roles are mostly theoretical (for example, every node is a Learner and most of the nodes are acceptors)
- Now, suppose the client makes a request to the system; the end goal of this action is to update every node's state (for example, the Input registry of the state machines) with the new client request
- In order to do that, firstly, a leader must be chosen to serve this particular request, also called a **Proposer**; then the set of **Acceptors** must be decided (usually the rest of the nodes have this role); then, there are 2 main phases:
  - **Establishing the state**: the **Proposer** sends an identifier (let's say $n$) for the request greater than all the previously known identifiers (by him) to a **Quorum of Acceptors** (must be a majority of acceptors) to let them know its "authority"; the chosen acceptors have 2 choices: either respond to that and "promise" through a request to the proposer they won't take any request with identifier less then or equal to $n$, where $n$ is the biggest number whose value they accepted so far (together with the current **accepted value**, if any, to maintain the safety of the protocol), or they can ignore it; the ignorance of the request is optional and is more an optimization: the proposer can halt after phase 1 when it noticed that at least one acceptor ignored the request, in which case it will increase the identifier or stop (since there is another leader in the system with greater authority); that's why the acceptor promises the proposer to ignore these types of requests; additionally, when a proposer failed to propagate its state in favor of another one already elected for slot $i$, that value is re-queued for slot $i + 1$
  - **State commitment**: the proposer gathers the necessary information from the acceptors and decides which value to propagate: either its value, or the majority's value, accepted previously for the slot $i$; after that, the proposer just broadcasts the chosen value in the set of acceptors and, further, to learners


## Multi-Paxos (practical)

- An important observation in the **Basic Paxos** is when there are multiple leaders, as there can be a single state to be propagated to the set of acceptors, there will surely be conflicts, where all but one intial values will be left aside
- Moreover, there can be a case where multiple leaders can conflict with each other in the Accept phase (that is, after the initial promise and before the current accept an overlapped acceptor promised to another leader with greater proposal number; then the first proposer will increase its identifier and then will be a conflict on the other side and so on)
- To solve this issue, **Multi-Paxos** version of the algorithm implements 2 fixes:
  - firstly, a single Proposer that is elected once in a long time, that is tasked to propagate that specific state in the system; in this way, there won't be any conflicts and there could be some optimizations to the algorithm involving some caching in order to skip the phase 1
  - secondly, to organize better the communication between client and system, a requests queue is created for the proposer; thus, further optimizations regarding batching multiple instructions from the queue can be done by the leader
- Still, the Paxos mechanisms are still useful when the only leader fails: in this situation, a new leader must be elected, a procedure which is done in the same manner as synchronizing a state in the **Basic Paxos** protocol
- Moreover, when the failed leader recovers and still believes it's the leader, the protocols will quickly let it know it has lost its authority (since at least one will return NAK)
- Now, regarding the acceptors, when some of them fail (based on the starting assumptions of SMR, at most $F$), then they shouldn't be allowed to vote since the quorum can reach a state where proposing outdated values; thus, a synchronization procedure was implemented, that states that any outdated acceptor, at recovery, must communicate with the leader to check the current state of the system and update its own state if necessary
- The learners broadcasting may seem trivial at first, but it's pretty problematic in such a fault-tolerant system: if each acceptor sends the verdict to each learner is computationally expensive; if each acceptor sends the result to a single leader-learner than it could fail; the solution is somewhere in the middle, that is there is a set of learner who communicate with all the acceptors and further send the results to the other learners: in this way, even though the communication is more complex, the system is failure-tolerant
- One last detail is how to synchronize the identifiers within the system but to also maintain the failure-tolerance property; a solution to this problem is to make sure the sets of identifiers are pairwise distinct between the nodes and the following identifier is greater then the previous identifiers on any other node (for example, one node starts at 1, another at 2 etc. and the step is $n$ - the number of nodes in the system; also, the identifiers are increased with the number of rounds passed); in this way, there won't be any issues with synchronization and, looking at the next empty slot, the procedure of finding the next available is trivial from each node (when some node becomes the leader)
- As a final note, when the single leader is elected by a majority of nodes, only the phase 2 is sufficient to be executed since the leader can store / cache the needed information of the system and act accordingly, all of this until the leader fails and a new one must be reelected
- The above observation gives the system the optimality property: there is no other way to decide and transfer the state across a distributed system more efficient that how the system works between the elections

