import zmq
import sys
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import List
from uuid import uuid4, UUID
from heapq import heappush, heappop


class Status(Enum):
    DOWN = 0,
    ELECTION = 1,
    REORGANIZATION = 2,
    NORMAL = 3

class MessageTypes(Enum):
    ARE_YOU_COORDINATOR = 0,
    TIMEOUT = 1,
    CHECK = 2,
    ARE_YOU_THERE = 3,
    I_AM_HERE = 4,
    MERGE = 5,
    INVITATION = 6,
    ACCEPT = 7,
    ACCEPTED = 8,
    FINISH_MERGE = 9,
    READY = 10


@dataclass(order=True)
class Timeout():
    id: UUID = field(compare=False)
    receiver_id: int = field(compare=False)
    type: MessageTypes = field(compare=False)
    data: any = field(compare=False)
    expiration_date: datetime

@dataclass
class Message():
    type: MessageTypes
    sender: int
    timeout: Timeout = Timeout()
    payload: any

@dataclass
class State():
    id: int
    port: int
    group_id: tuple[int, int]
    status: Status
    neighbour_ids: List[int]
    group_counter: int = 0
    outboxes: dict
    coordinator_id: int
    participants: List[int]
    timeout_pool: List[Timeout] = []
    timeout_data: dict = {}
    timeout_id: UUID


BASE_PORT = 8000
TIMEOUT = 3
MERGE_TIMEOUT = 7
COORDINATOR_COMMUNICATION = 10


def timeout_end(k: float = TIMEOUT) -> datetime:
    return datetime.now() + timedelta(seconds=k)


def set_timeout(timeout_id: UUID, participant: int, message_type: MessageTypes, 
                payload: any = None, default_timeout_data: any = False, expiration=TIMEOUT, send_message = True):
    global state
    current_timeout = Timeout(
        id=timeout_id,
        receiver_id=participant,
        type=message_type,
        expiration_date=timeout_end(expiration)
    )

    if send_message:
        message = Message(
            type=message_type,
            sender=state.id,
            timeout=current_timeout,
            payload=payload
        )
        state.outboxes[participant].send_json(message)

    heappush(state.timeout_pool, current_timeout)
    state.timeout_data[timeout_id] = default_timeout_data
    return timeout_id


if len(sys.argv) < 2:
    print("Usage: python node.py <ID> [NEIGHBOURING_PORTS...]")
    exit(0)


state: State = State(
    id=int(sys.argv[1]),
    port=BASE_PORT + int(sys.argv[1]),
    coordinator_id=int(sys.argv[1]),
    group_id=(int(sys.argv[1]), 0),
    status=Status.NORMAL,
    neighbour_ids=[int(neighbour_id) for neighbour_id in sys.argv[2:]],
    outboxes={},
    participants=[int(sys.argv[1])],
    timeout_id=set_timeout(uuid4(), int(sys.argv[1]), MessageTypes.TIMEOUT, None, False, COORDINATOR_COMMUNICATION, False)
)


def check():
    global state
    if state.status == Status.NORMAL and state.id == state.coordinator_id:
        state.timeout_data[timeout_id] = []
        timeout_id = uuid4()

        for neighbour_id in state.neighbour_ids:
            if state.id >= neighbour_id:
                continue
            
            election_msg = Message(
                type=MessageTypes.ARE_YOU_COORDINATOR,
                sender=state.id,
                payload=timeout_id
            )
            state.outboxes[neighbour_id].send_json(election_msg)
        
        current_timeout = Timeout(
            id=timeout_id,
            receiver_id=neighbour_id,
            type=MessageTypes.ARE_YOU_COORDINATOR,
            expiration_date=timeout_end()
        )
        heappush(state.timeout_pool, current_timeout)
        state.timeout_data[timeout_id] = []


def are_you_there(group_id: tuple[int, int], sender_id: int):
    global state
    if state.id == group_id[0] and state.coordinator_id == state.id and sender_id in state.participants:
        message = Message(
            type=MessageTypes.I_AM_HERE,
            sender=state.id
        )
        state.outboxes[sender_id].send_json(message)


def I_am_here(timeout: Timeout):
    global state
    if timeout.expiration_date < datetime.now():
        print(f"Timeout for node {timeout.receiver_id} expired before it sent back the response")
        return
    state.timeout_data[timeout.id] = True


def recovery():
    global state
    state.status = Status.ELECTION
    state.group_counter += 1
    state.group_id = (state.id, state.group_counter)
    state.coordinator_id = state.id
    state.participants = []
    state.status = Status.REORGANIZATION
    # Add a task here
    state.status = Status.NORMAL


def timeout():
    global state
    if state.id == state.coordinator_id:
        return

    set_timeout(uuid4(), state.coordinator_id, MessageTypes.ARE_YOU_THERE, state.group_id)


def merge(coordinator_set: List[int]):
    global state
    state.status = Status.ELECTION
    state.group_counter += 1
    state.group_id = (state.id, state.group_counter)
    state.coordinator_id = state.id
    old_participants = state.participants
    state.participants = []
    
    for participant in coordinator_set:
        state.outboxes[participant].send_json(Message(
            type=MessageTypes.INVITATION,
            sender=state.id,
            payload={
                "group": state.group_id,
                "expiration_date": timeout_end()
            }
        ))

    for participant in old_participants:
        state.outboxes[participant].send_json(Message(
            type=MessageTypes.INVITATION,
            sender=state.id,
            payload={
                "group": state.group_id,
                "expiration_date": timeout_end()
            }
        ))
    
    set_timeout(uuid4(), state.id, MessageTypes.FINISH_MERGE, None, False, MERGE_TIMEOUT, False)


def finish_merge():
    global state
    state.status = Status.REORGANIZATION

    timeout_id = uuid4()
    payload = {
        "group": state.group_id,
        "task": "hellowww",
        "timeout_id": timeout_id
    }
    for participant in state.participants:
        set_timeout(timeout_id, participant, MessageTypes.READY, payload)
    
    state.status = Status.NORMAL


def ready(group_number: tuple[int, int], task_description: str, timeout_id: UUID):
    global state
    if state.status == Status.REORGANIZATION and state.group_id[0] == group_number[0]:
        print(task_description)
        state.status = Status.NORMAL
    
    state.timeout_data[timeout_id] = True


def invitation(sender: int, group_number: tuple[int, int]):
    global state
    if state.status != Status.NORMAL:
        return
    
    old_coordinator = state.coordinator_id
    participants = state.participants
    state.status = Status.ELECTION
    state.coordinator_id = sender
    state.group_id = group_number

    if old_coordinator == state.id:
        for participant in participants:
            state.outboxes[participant].send_json(Message(
                type=MessageTypes.INVITATION,
                sender=state.id,
                payload={
                    "group": state.group_id,
                    "expiration_date": timeout_end()
                }
            ))
    
    set_timeout(uuid4(), state.id, MessageTypes.ACCEPT, {
        "group": state.group_id,
        "expiration_date": timeout_end()
    })


def accept(sender: int, group_number: tuple[int, int], timeout_id: UUID):
    global state
    if state.status == Status.ELECTION and state.group_id == group_number and state.coordinator_id == state.id:
        state.participants.append(sender)
    
    state.timeout_data[timeout_id] = True


def accepted():
    global state
    state.status = Status.REORGANIZATION


def run_node():
    global state
    context = zmq.Context()

    inbox = context.socket(zmq.ROUTER)
    inbox.setsockopt_string(zmq.IDENTITY, state.id)
    inbox.bind(f"tcp://*:{state.port}")

    for neighbour in state.neighbour_ids:
        dealer = context.socket(zmq.DEALER)
        dealer.setsockopt_string(zmq.IDENTITY, state.id)
        dealer.connect(f"tcp://localhost:{state.port}")
        state.outboxes[neighbour] = dealer

    poller = zmq.Poller()
    poller.register(inbox, zmq.POLLIN)

    print(f"Node {state.id} started on port {state.port}...")

    try:
        while True:
            # Poll for 100ms
            socks = dict(poller.poll(100))

            if inbox in socks:
                sender_id_bytes, payload_bytes = inbox.recv_multipart()
    
                data: Message = json.loads(payload_bytes.decode())
                sender_id = sender_id_bytes.decode()

                if sender_id == state.coordinator_id:
                    state.timeout_data[state.timeout_id] = True

                match data.type:
                    case MessageTypes.ARE_YOU_COORDINATOR:
                        print(f"Are you coordinator question asked by {sender_id}")
                        if state.coordinator_id == state.id and state.status == Status.NORMAL:
                            state.timeout_data[data.timeout.id] = \
                                state.timeout_data.get(data.timeout.id, []) + [data.timeout.receiver_id]
                    case MessageTypes.CHECK:
                        check()
                    case MessageTypes.ARE_YOU_THERE:
                        are_you_there(data.payload, state.id)
                    case MessageTypes.I_AM_HERE:
                        I_am_here(data.timeout)
                    case MessageTypes.INVITATION:
                        if data.payload["expiration_date"] > datetime.now():
                            invitation(data.sender, data.payload)
                    case MessageTypes.ACCEPT:
                        accept(data.sender, data.payload["group"], data.timeout.id)
                    case MessageTypes.READY:
                        ready(data.payload["group"], data.payload["task"], data.payload["timeout_id"])
                    case _:
                        print(f"No known message type {data.type}")

            while len(state.timeout_pool) > 0 and state.timeout_pool[0].expiration_date < datetime.now():
                last_job = heappop(state.timeout_pool)
                match last_job.type:
                    case MessageTypes.ARE_YOU_THERE:
                        if not state.timeout_data[last_job.id]:
                            recovery()
                        del state.timeout_data[last_job.id]
                    case MessageTypes.ARE_YOU_COORDINATOR:
                        max_node = max(state.timeout_data[last_job.id])
                        heappush(state.timeout_pool, Timeout(
                            id=uuid4(),
                            receiver_id=state.id,
                            type=MessageTypes.MERGE,
                            expiration_date=timeout_end(max(0, max_node - state.id) / 10),
                            data=state.timeout_data[last_job.id]
                        ))
                        del state.timeout_data[last_job.id]
                    case MessageTypes.MERGE:
                        merge(last_job.data)
                    case MessageTypes.ACCEPT:
                        if state.timeout_data[last_job.id]:
                            accepted()
                        else:
                            recovery()
                        del state.timeout_data[last_job.id]
                    case MessageTypes.FINISH_MERGE:
                        finish_merge()
                        del state.timeout_data[last_job.id]
                    case MessageTypes.READY:
                        if not state.timeout_data[last_job.id]:
                            recovery()
                        del state.timeout_data[last_job.id]
                    case MessageTypes.TIMEOUT:
                        if not state.timeout_data[last_job.id]:
                            timeout()
                        del state.timeout_data[last_job.id]
                        state.timeout_id = set_timeout(uuid4(), state.id, MessageTypes.TIMEOUT, 
                                                       None, False, COORDINATOR_COMMUNICATION, False)
                    case _:
                        print(f"No known Message Type in timeout pool {last_job.type}")
                        
            time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\nNode {state.id} shutting down...")
    finally:
        inbox.close()
        for s in state.outboxes.values():
            s.close()
        context.term()


run_node()