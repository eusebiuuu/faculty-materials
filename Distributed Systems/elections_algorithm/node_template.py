import os
import sys
import time
import json
import pickle
import signal
import logging
from zmq import Context, Socket, Poller, DEALER, ROUTER, POLLIN, IDENTITY
from datetime import datetime, timedelta
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import List
from uuid import uuid4
from heapq import heappush, heappop

#---------------Classes-------------------

class Status(Enum):
    DOWN = 0
    ELECTION = 1
    REORGANIZATION = 2
    NORMAL = 3

class MessageTypes(Enum):
    ARE_YOU_COORDINATOR = 0
    TIMEOUT = 1
    CHECK = 2
    ARE_YOU_THERE = 3
    I_AM_HERE = 4
    MERGE = 5
    INVITATION = 6
    ACCEPT = 7
    ACCEPTED = 8
    REORGANIZATION_MERGE = 9
    READY = 10
    COORDINATOR_STATE = 11
    FINISH_MERGE = 12
    SHOW_STATE = 13


@dataclass(order=True)
class Timeout():
    id: str = field(compare=False)
    type: int = field(compare=False)
    expiration_date: datetime


@dataclass
class Message():
    type: int
    sender: int
    payload: any = None
    expiration: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "sender": self.sender,
            "payload": self.payload,
            "expiration": self.expiration.isoformat() if self.expiration else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            type=MessageTypes(data["type"]),
            sender=data["sender"],
            payload=data.get("payload"),
            expiration=datetime.fromisoformat(data["expiration"]) if data["expiration"] else None
        )

@dataclass
class State():
    id: int
    port: int
    group_id: tuple[int, int]
    status: int
    neighbour_ids: List[int]
    outboxes: dict
    coordinator_id: int
    participants: List[int]
    timeout_id: str | None = field(default=None)
    group_counter: int = 0
    timeout_pool: List[Timeout] | None = None
    timeout_data: dict | None = field(default=None)


#---------------Constants-------------------

BASE_PORT = 8000
TIMEOUT = 3
MERGE_TIMEOUT = 7
COORDINATOR_COMMUNICATION = 20
INVALID = -1
TEMP_STATE_SAVING_LOCATION = "temp_state_checkpoint.pkl"
STATE_FILENAME = "state_checkpoint.pkl"

#---------------Logging Setup-------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

#---------------Utils-------------------

def save_state(state: State):
    # We create a copy of the state but empty out the 'outboxes' 
    # because sockets cannot be pickled/serialized.
    state_to_save = replace(state, outboxes={})
    
    with open(TEMP_STATE_SAVING_LOCATION, "wb") as f:
        pickle.dump(state_to_save, f)
        # Ensure data is physically written to disk
        os.fsync(f.fileno())
    
    os.replace(TEMP_STATE_SAVING_LOCATION, STATE_FILENAME)


def load_state() -> State:
    if not os.path.exists(STATE_FILENAME):
        return None
        
    with open(STATE_FILENAME, "rb") as f:
        state = pickle.load(f)
    
    state.status = Status.NORMAL.value

    return state


def initialize_outboxes(state: State, context: Context[Socket[bytes]]):
    inbox = context.socket(ROUTER)
    inbox.setsockopt_string(IDENTITY, str(state.id))
    inbox.bind(f"tcp://*:{state.port}")

    for neighbour in state.neighbour_ids:
        dealer = context.socket(DEALER)
        dealer.setsockopt_string(IDENTITY, str(state.id))
        dealer.connect(f"tcp://localhost:{BASE_PORT + neighbour}")
        state.outboxes[neighbour] = dealer

    state.timeout_pool = []
    state.timeout_data = {}
    state.timeout_id = set_timeout(str(uuid4()), int(sys.argv[1]),
                                   MessageTypes.TIMEOUT, None, False, COORDINATOR_COMMUNICATION, False)
    return inbox


def timeout_end(k: float = TIMEOUT) -> datetime:
    return datetime.now() + timedelta(seconds=k)


def send_message(receiver: int, message: Message):
    global state
    try:
        state.outboxes[receiver].send_json(message.to_dict())
    except Exception:
        logging.info(f"Cannot connect to node {receiver}")


def set_timeout(timeout_id: str, participant: int, message_type: MessageTypes, 
                payload: any = None, default_timeout_data: any = False, expiration=TIMEOUT, send_message_flag = True):
    global state

    state.timeout_data[timeout_id] = default_timeout_data
    
    current_timeout = Timeout(
        id=timeout_id,
        type=message_type.value,
        expiration_date=timeout_end(expiration)
    )

    if send_message_flag:
        message = Message(
            type=message_type.value,
            sender=state.id,
            expiration=timeout_end(expiration),
            payload=payload
        )
        # logging.info(f"Sender: {state.id}; Type: {message_type.value}")
        
        send_message(participant, message)

    heappush(state.timeout_pool, current_timeout)
    return timeout_id

#---------------Global Variables Initialization-------------------

context = Context()

neighbours = sys.argv[2:] if len(sys.argv) >= 2 else []

state: State | None = None

try:
    state = load_state()
finally:
    if state == None:
        state = State(
            id=int(sys.argv[1]),
            port=BASE_PORT + int(sys.argv[1]),
            coordinator_id=int(sys.argv[1]),
            group_id=(int(sys.argv[1]), 0),
            status=Status.NORMAL.value,
            neighbour_ids=[int(neighbour_id) for neighbour_id in neighbours] + [int(sys.argv[1])],
            outboxes={},
            participants=[],
            timeout_id=None
        )

inbox = initialize_outboxes(state, context)

#---------------Consensus Algorithm-------------------

def check():
    logging.info("CHECK called")
    global state
    if state.status == Status.NORMAL.value and state.id == state.coordinator_id:
        timeout_id = str(uuid4())
        state.timeout_data[timeout_id] = []

        for neighbour_id in state.neighbour_ids:
            election_msg = Message(
                type=MessageTypes.ARE_YOU_COORDINATOR.value,
                sender=state.id,
                payload=timeout_id
            )
            # logging.info(f"Sender: {state.id}; Type: {MessageTypes.ARE_YOU_COORDINATOR.value}; Neighbour {neighbour_id}")
            # logging.info(election_msg.to_dict())
            send_message(neighbour_id, election_msg)
        
        current_timeout = Timeout(
            id=timeout_id,
            type=MessageTypes.ARE_YOU_COORDINATOR.value,
            expiration_date=timeout_end()
        )
        heappush(state.timeout_pool, current_timeout)


def coordinator_state(timeout_id, sender: int):
    global state
    logging.info("COORDINATOR STATE called")
    message = Message(
        type=MessageTypes.COORDINATOR_STATE.value,
        sender=state.id,
        payload=timeout_id
    )    
    send_message(sender, message)


def are_you_there(group_id: tuple[int, int], timeout_id: str, sender_id: int):
    logging.info(f"ARE YOU THERE called - {sender_id} sent the message")
    global state
    if state.id == group_id[0] and state.coordinator_id == state.id and sender_id in state.participants:
        message = Message(
            type=MessageTypes.I_AM_HERE.value,
            sender=state.id,
            payload=timeout_id
        )
        logging.info(f"Sender: {state.id}; Type: {MessageTypes.I_AM_HERE.value}")
        send_message(sender_id, message)


def I_am_here(timeout_id: str):
    logging.info("I AM HERE called")
    global state
    state.timeout_data[timeout_id] = True


def show_state():
    global state
    logging.info("=" * 50)
    logging.info(f"Coordinator: {state.coordinator_id}")
    logging.info(f"Status: {state.status}")
    logging.info(f"Participants: {state.participants}")
    logging.info("=" * 50)


def recovery():
    logging.info("RECOVERY called")
    global state
    state.status = Status.ELECTION.value
    state.group_counter += 1
    state.group_id = (state.id, state.group_counter)
    state.coordinator_id = state.id
    state.participants = []
    state.status = Status.REORGANIZATION.value
    state.status = Status.NORMAL.value


def timeout():
    global state
    logging.info(f"TIMEOUT called when status is {state.status}")
    if state.id == state.coordinator_id:
        return

    timeout_id = str(uuid4())
    state.timeout_data[timeout_id] = False

    payload = {
        "group": state.group_id,
        "timeout": timeout_id,
    }
    set_timeout(timeout_id, state.coordinator_id, MessageTypes.ARE_YOU_THERE, payload)


def merge(coordinator_set: List[int]):
    logging.info(f"MERGE called with coordinator set {coordinator_set}")
    global state

    if state.status != Status.NORMAL.value:
        return
    
    state.status = Status.ELECTION.value
    state.group_counter += 1
    state.group_id = (state.id, state.group_counter)
    state.coordinator_id = state.id
    old_participants = state.participants
    state.participants = []

    timeout_id = str(uuid4())
    
    for participant in coordinator_set:
        # logging.info(f"Sender: {state.id}; Receiver: {participant}")
        send_message(participant, Message(
            type=MessageTypes.INVITATION.value,
            sender=state.id,
            expiration=timeout_end(),
            payload=state.group_id,
        ))

    for participant in old_participants:
        # logging.info(f"Sender: {state.id}; Receiver: {participant}")
        send_message(participant, Message(
            type=MessageTypes.INVITATION.value,
            sender=state.id,
            expiration=timeout_end(),
            payload=state.group_id,
        ))
    
    set_timeout(timeout_id, state.id, MessageTypes.REORGANIZATION_MERGE, None, False, MERGE_TIMEOUT, False)


def reorganization_merge():
    global state
    logging.info(f"REORGANIZATION MERGE called for participants {state.participants} and status {state.status}")
    state.status = Status.REORGANIZATION.value

    timeout_id = str(uuid4())
    payload = {
        "group": state.group_id,
        "task": "Task Completed!",
        "timeout_id": timeout_id
    }

    for participant in state.participants:
        send_message(participant, Message(
            type=MessageTypes.READY.value,
            sender=state.id,
            payload=payload
        ))
    
    set_timeout(timeout_id, state.id, MessageTypes.FINISH_MERGE, None, True, TIMEOUT, False)
    

def finish_merge():
    logging.info("FINISH MERGE called")
    global state
    state.status = Status.NORMAL.value


def ready(group_number: tuple[int, int], task_description: str, sender: str):
    global state
    logging.info(f"Ready called by {sender} when status is {state.status}; {state.group_id[0]} vs {group_number[0]}")
    if state.status == Status.REORGANIZATION.value and state.group_id[0] == group_number[0]:
        logging.info(task_description)
        state.status = Status.NORMAL.value


def invitation(sender: int, group_number: tuple[int, int]):
    global state
    logging.info(f"INVITATION sent by {sender} when status is {state.status}")

    if state.status != Status.NORMAL.value:
        return
    
    old_coordinator = state.coordinator_id
    participants = state.participants
    state.status = Status.ELECTION.value
    state.coordinator_id = sender
    state.group_id = group_number

    logging.info(f"New coordinator: {state.coordinator_id}")

    timeout_id = str(uuid4())
    state.timeout_data[timeout_id] = {
        "group": state.group_id,
        "sender": sender
    }

    if old_coordinator == state.id:
        for participant in participants:
            # logging.info(f"Sender: {state.id}; Type: {MessageTypes.INVITATION.value}")
            send_message(participant, Message(
                type=MessageTypes.INVITATION.value,
                sender=state.id,
                expiration=timeout_end(),
                payload=state.group_id,
            ))
    
    current_timeout = Timeout(
        id=timeout_id,
        type=MessageTypes.INVITATION.value,
        expiration_date=timeout_end(),
    )
    heappush(state.timeout_pool, current_timeout)


def send_accept(sender: int, group_number: tuple[int, int]):
    logging.info(f"SENDING ACCEPT to {sender}")
    global state

    timeout_id = str(uuid4())
    payload = {
        "sender": sender,
        "group": group_number,
        "timeout": timeout_id
    }

    set_timeout(timeout_id, sender, MessageTypes.ACCEPT, payload)


def accept(sender: int, group_number: tuple[int, int], timeout_id: str):
    logging.info(f"ACCEPT called: {sender} accepted invitation")
    global state

    logging.info(f"State: {state.status}; Group: {state.group_id[0]} vs {group_number[0]}; Coordinator: {state.coordinator_id}")
    if state.status == Status.ELECTION.value and state.group_id[0] == group_number[0] and state.coordinator_id == state.id:
        state.participants.append(sender)
        logging.info(f"ACCEPT: participant saved")

    send_message(sender, Message(
        type=MessageTypes.ACCEPTED.value,
        sender=state.id,
        payload=timeout_id
    ))


def accepted():
    logging.info("ACCEPTED called")
    global state
    if state.status == Status.NORMAL:
        return
    
    state.status = Status.REORGANIZATION.value

#---------------Node Runner-------------------

def run_node():
    logging.info("Running node...")
    global state
    global context
    global inbox

    poller = Poller()
    poller.register(inbox, POLLIN)

    logging.info(f"Node {state.id} started on port {state.port}...")

    while True:
        # Poll for 100ms
        socks = dict(poller.poll(100))

        if inbox in socks:
            sender_id_bytes, payload_bytes = inbox.recv_multipart()

            dict_data: dict = json.loads(payload_bytes.decode())
            # logging.info(dict_data)
            data = Message.from_dict(dict_data)
            sender_id = int(sender_id_bytes.decode())

            if sender_id == state.coordinator_id:
                state.timeout_data[state.timeout_id] = True

            match data.type:
                case MessageTypes.ARE_YOU_COORDINATOR:
                    logging.info(f"Are you coordinator question asked by {sender_id}")
                    if state.coordinator_id == state.id and state.status == Status.NORMAL.value:
                        coordinator_state(data.payload, sender_id)
                case MessageTypes.CHECK:
                    check()
                case MessageTypes.COORDINATOR_STATE:
                    if state.timeout_data.get(data.payload, INVALID) != INVALID:
                        logging.info(f"Updating coordinator set with node {sender_id}")
                        state.timeout_data[data.payload] = state.timeout_data.get(data.payload, []) + [sender_id]
                case MessageTypes.ARE_YOU_THERE:
                    are_you_there(data.payload["group"], data.payload["timeout"], sender_id)
                case MessageTypes.ACCEPTED:
                    state.timeout_data[data.payload] = True
                case MessageTypes.I_AM_HERE:
                    I_am_here(data.payload)
                case MessageTypes.INVITATION:
                    if data.expiration > datetime.now():
                        logging.info(f"Invitation from node {sender_id} received")
                        invitation(data.sender, data.payload)
                case MessageTypes.ACCEPT:
                    accept(sender_id, data.payload["group"], data.payload["timeout"])
                case MessageTypes.SHOW_STATE:
                    show_state()
                case MessageTypes.READY:
                    ready(data.payload["group"], data.payload["task"], sender_id)
                case _:
                    logging.info(f"No known message type {data.type}")

        while len(state.timeout_pool) > 0 and state.timeout_pool[0].expiration_date < datetime.now():
            last_job = heappop(state.timeout_pool)
            last_job_type = MessageTypes(last_job.type)
            match last_job_type:
                case MessageTypes.ARE_YOU_THERE:
                    if not state.timeout_data[last_job.id]:
                        recovery()
                    state.timeout_data[last_job.id] = None
                case MessageTypes.ARE_YOU_COORDINATOR:
                    logging.info(f"ARE YOU COORDINATOR timeout finished with length {len(state.timeout_data[last_job.id])}")
                    if len(state.timeout_data[last_job.id]) > 0:
                        max_node = max(state.timeout_data[last_job.id])
                        logging.info("Setting merge timeout...")
                        current_timeout_id = str(uuid4())
                        heappush(state.timeout_pool, Timeout(
                            id=current_timeout_id,
                            type=MessageTypes.MERGE,
                            expiration_date=timeout_end(max(0, max_node - state.id)),
                            # expiration_date=timeout_end(),
                        ))
                        state.timeout_data[current_timeout_id] = state.timeout_data[last_job.id]
                    state.timeout_data[last_job.id] = None
                case MessageTypes.MERGE:
                    merge(state.timeout_data[last_job.id])
                case MessageTypes.ACCEPT:
                    if state.timeout_data.get(last_job.id, None):
                        accepted()
                    else:
                        recovery()
                    state.timeout_data[last_job.id] = None
                case MessageTypes.REORGANIZATION_MERGE:
                    reorganization_merge()
                    state.timeout_data[last_job.id] = None
                case MessageTypes.FINISH_MERGE:
                    state.timeout_data[last_job.id] = None
                    finish_merge()
                case MessageTypes.INVITATION:
                    last_job_data = state.timeout_data[last_job.id]
                    send_accept(last_job_data["sender"], last_job_data["group"])
                case MessageTypes.READY:
                    if not state.timeout_data[last_job.id]:
                        recovery()
                    state.timeout_data[last_job.id] = None
                case MessageTypes.TIMEOUT:
                    if not state.timeout_data[last_job.id]:
                        timeout()
                    state.timeout_data[last_job.id] = None
                    
                    state.timeout_id = set_timeout(str(uuid4()), state.id, MessageTypes.TIMEOUT, 
                                                    None, False, COORDINATOR_COMMUNICATION, False)
                case _:
                    logging.info(f"No known Message Type in timeout pool: {last_job.type}")
                
            if state.timeout_data.get(last_job.id, INVALID) != INVALID:
                del state.timeout_data[last_job.id]
                    
        save_state(state)
        time.sleep(0.1)
        # break


def handle_sigterm(signum, frame):
    global state
    global context
    global inbox

    show_state()

    inbox.close()
    for s in state.outboxes.values():
        s.close()
    context.term()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

run_node()

