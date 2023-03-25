import socket
from gnn_a2c_lm.graph_neural_network_advantage_actor_critic_lagrange_multiplier import *

def readState(line, n):
    line = line.split(' ')
    b_d = []
    for i in range(n):
        b_d.append([(float)(line[2 * i]), (float)(line[2 * i + 1])])
    pt_sc = [(float)(line[2 * n]), (float)(line[2 * n + 1])]
    state = (b_d, pt_sc)
    return state

class Allocator:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(('localhost',9999))
    def allocate(self,n):
        agent = Agent(n=n, lr_critic=0, lr_actor=0, batch_size=0, target_update=0, gamma=0,
                      path=r"F:\PythonWorkspace\nonlinear_programming\gnn_a2c_lm\GALModel.pkl")
        while(True):
            data = self.sock.recv(1024)
            data = str(data, 'UTF-8')
            print(data)
            state = readState(data, n)
            action, _ = agent.interacte(state)
            policy = agent.action_to_policy(state, action)
            saveContent = ""
            savePolicy = policy[0] + policy[1] + policy[2]
            for i in range(len(savePolicy)):
                saveContent += str(savePolicy[i])
                if i != len(savePolicy) - 1:
                    saveContent += " "
            self.sock.send(saveContent.encode('utf-8'))

def main():
    Allocator().allocate(3)

if __name__ == "__main__":
    main()