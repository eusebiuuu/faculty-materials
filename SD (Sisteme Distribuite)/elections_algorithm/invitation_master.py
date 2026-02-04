import zmq
import sys
import os
import time
import glob
import shutil
import logging
from subprocess import Popen, STDOUT
import sys
from enum import Enum
from dataclasses import dataclass
from typing import List
from datetime import datetime, timedelta
from heapq import heappush, heappop
from threading import Thread, Event


MY_ID = 0
MY_PORT = 8000
SOURCE_FILE = "node_template.py"
INPUT_FILE = "topology.txt"
OUTPUT_FILENAME = "master_output.log"
PERIOD = 20


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
            "timeout": None,
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
class Request():
    type: int
    expiration: datetime


stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_formatter = logging.Formatter(
    "[%(levelname)s] %(message)s"
)
stdout_handler.setFormatter(stdout_formatter)

file_handler = logging.FileHandler(OUTPUT_FILENAME)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s %(name)s %(levelname)s %(message)s"
)
file_handler.setFormatter(file_formatter)

master_logger = logging.getLogger("master logger")
master_logger.setLevel(logging.DEBUG)
master_logger.addHandler(stdout_handler)
master_logger.addHandler(file_handler)


def timeout_end(k: float = PERIOD) -> datetime:
    return datetime.now() + timedelta(seconds=k)


input_lines = []
with open(INPUT_FILE) as topology_file:
    input_lines = [line.strip() for line in topology_file]

# Base directory for the test run
workspace = "dist_test_run"
if os.path.exists(workspace):
    shutil.rmtree(workspace)
os.makedirs(workspace)

processes: List[Popen | None] = [None]
requests: List[Request] = []
context = zmq.Context()
outboxes = {}
num_nodes = int(input_lines[0])
master_logger.debug(f"Starting {num_nodes} nodes...")

dealer = context.socket(zmq.DEALER)
dealer.setsockopt_string(zmq.IDENTITY, str(MY_ID))
master_logger.debug(f"Master {MY_ID} started on port {MY_PORT}...")


for node_id in range(1, num_nodes + 1):
    node_dir = os.path.join(workspace, f"node_{node_id}")
    os.makedirs(node_dir)

    # 1. Copy the logic file into the directory
    target_file = os.path.join(node_dir, "node.py")
    shutil.copy(SOURCE_FILE, target_file)

    # 2. Prepare the arguments for this specific node
    # Argv: [script, ID, my_port, neighbor_port1, neighbor_port2...]
    my_port = MY_PORT + node_id
    neighbors = input_lines[node_id].split(" ")
    
    cmd = [sys.executable, "-u", "node.py", str(node_id)] + neighbors

    # 3. Execute the process
    # We use cwd (current working directory) to ensure the node runs inside its folder
    p = Popen(
        cmd,
        cwd=node_dir,
        stdout=open(os.path.join(node_dir, "output.log"), "w"),
        stderr=STDOUT
    )
    processes.append(p)
    master_logger.debug(f"  > Spawned Node {node_id} on port {my_port} (PID: {p.pid})")
    dealer.connect(f"tcp://localhost:{my_port}")
    outboxes[node_id] = dealer

master_logger.info("All nodes are running. Logs are being written to their respective folders.")
master_logger.info("Write 'Exit' in terminal if you want to end the program and terminate all the nodes gracefully")


def send_check(receiver_id: int):
    global outboxes
    master_logger.debug(f"Send check to node {receiver_id}...")
    message: Message = Message(
        type=MessageTypes.CHECK.value,
        sender=MY_ID
    )
    outboxes[receiver_id].send_json(message.to_dict())


def show_states():
    global outboxes
    global num_nodes
    master_logger.debug(f"Check the output.log files for processes states")
    for i in range(1, num_nodes + 1):
        message: Message = Message(
            type=MessageTypes.SHOW_STATE.value,
            sender=MY_ID
        )
        outboxes[i].send_json(message.to_dict())


def crash_node(processes_list, node_id):
    """
    Simulates a hard crash of a specific node.
    processes_dict: a dictionary mapping 'A', 'B' -> Popen object
    """

    if processes_list[node_id]:
        proc = processes_list[node_id]
        master_logger.info(f"!!! CRASHING NODE {node_id} (PID {proc.pid}) !!!")
        
        proc.kill()
        
        proc.wait()
        processes_list[node_id] = None
    else:
        master_logger.info(f"Node {node_id} is not running.")


def reboot_node(node_id: int):
    global workspace
    global processes

    if processes[node_id]:
        master_logger.info(f"Node {node_id} is already up and running")
        return

    node_dir = os.path.join(workspace, f"node_{node_id}")
    cmd = [sys.executable, "-u", "node.py", str(node_id)]
    
    master_logger.info(f"--- Rebooting Node {node_id} ---")
    p = Popen(
        cmd,
        cwd=node_dir,
        stdout=open(os.path.join(node_dir, "output.log"), "a"),
        stderr=STDOUT
    )
    processes[node_id] = p


def run_system(stop_event: Event):
    global outboxes
    global workspace
    master_logger.info("Running system...")

    with open(OUTPUT_FILENAME, "r+") as f:
        f.seek(0)
        f.truncate()

    valid_task_ids = [1, 2, 3]

    while not stop_event.is_set():
        master_logger.info("Actions available:")
        master_logger.info("1 - Stop node")
        master_logger.info("2 - Reboot node")
        master_logger.info("3 - Show processes states")
        master_logger.info("4 - Exit and terminate all processes")

        action_str = input("Insert the action id to perform: ")

        if action_str.lower() == "exit" or action_str == str(max(valid_task_ids) + 1):
            stop_event.set()
            break
        
        try:
            action = int(action_str)
            if action not in valid_task_ids:
                raise Exception("Invalid action :(")
        except Exception:
            master_logger.info("Input must be an integer or 'Exit'")
            continue

        if action == 1:
            node_num = int(input(">> Node id: "))
            crash_node(processes, node_num)
        elif action == 2:
            node_num = int(input(">> Node id: "))
            reboot_node(node_num)
        elif action == 3:
            show_states()

    master_logger.debug(f"Node {MY_ID} shutting down...")
    master_logger.info("Terminating gracefully all nodes...")
        
    for s in outboxes.values():
        s.close()
    
    for p in processes:
        if p:
            p.terminate()
            p.wait(timeout=2)

    checkpoint_files = glob.glob(f"{workspace}/**/*.pkl", recursive=True)

    for f in checkpoint_files:
        os.remove(f)
    
    master_logger.debug("Done.")
    context.term()
        

def run_routines(stop_event: Event):
    global requests
    global num_nodes

    master_logger.debug("Running routines...")
    heappush(requests, Request(type=MessageTypes.CHECK, expiration=datetime.now()))

    while not stop_event.is_set():
        time.sleep(1)
        
        while len(requests) > 0 and requests[0].expiration < datetime.now():
            last_request: Request = heappop(requests)
            match last_request.type:
                case MessageTypes.CHECK:
                    for i in range(1, num_nodes + 1):
                        send_check(i)
                    heappush(requests, Request(type=MessageTypes.CHECK, expiration=timeout_end()))
    
    master_logger.debug(f"Shutting down the routines...")



stop_event = Event()

threads = [
    Thread(target=run_system, args=(stop_event,)),
    Thread(target=run_routines, args=(stop_event,))
]

for t in threads:
    t.start()

for t in threads:
    t.join()
