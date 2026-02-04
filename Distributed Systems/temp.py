from datetime import datetime, timedelta
from enum import Enum
import logging
import sys

print(datetime.now())
print(datetime.now() + timedelta(seconds=30))

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
    FINISH_MERGE = 9
    READY = 10

x: int = MessageTypes.CHECK.value

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     stream=sys.stdout
# )

# logging.info(x)

# logging.info(type(MessageTypes.ACCEPT.value))

# logging.info(type(x))

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_formatter = logging.Formatter(
    "[%(levelname)s] %(message)s"
)
stdout_handler.setFormatter(stdout_formatter)

file_handler = logging.FileHandler("temp.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s %(name)s %(levelname)s %(message)s"
)
file_handler.setFormatter(file_formatter)

master_logger = logging.getLogger("master logger")
master_logger.setLevel(logging.DEBUG)
master_logger.addHandler(stdout_handler)
master_logger.addHandler(file_handler)

master_logger.debug("heey")
master_logger.info("yooo")