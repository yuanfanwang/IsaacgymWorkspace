from env import Env
from policy import Policy

policy = Policy()
env = Env()

env.reset()

while True:
    action = 1
    env.step()
    

