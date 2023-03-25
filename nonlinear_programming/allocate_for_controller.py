import socket
from gnn_a2c_lm.graph_neural_network_advantage_actor_critic_lagrange_multiplier import *
import copy


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

    def allocateGALStrategy(self,n):
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

    def allocateRandomStrategy(self,n):
        while(True):
            data = self.sock.recv(1024)
            data = str(data, 'UTF-8')
            print(data)
            state = readState(data, n)
            action = []
            for j in range(n):
                action.append(round(random.random()))
            for j in range(n):
                action.append(random.random() * 4 + 1)
            for j in range(n):
                if action[j] == 1:
                    action.append(random.random() * 4 + 1)
                else:
                    action.append(0)
            sum_w = sum(action[n:2 * n])
            sum_b = sum(action[2 * n:3 * n])
            for j in range(n):
                action[n + j] = action[n + j] / sum_w * 7
                if sum_b != 0:
                    action[2 * n + j] = action[2 * n + j] / sum_b * 10

            content = ""
            for i in range(3 * n):
                content += str(action[i])
                if i != 3 * n - 1:
                    content += " "
            self.sock.send(content.encode('utf-8'))

    def allocateRandomEquipartitionStrategy(self,n):
        while(True):
            data = self.sock.recv(1024)
            data = str(data, 'UTF-8')
            print(data)
            state = readState(data, n)
            action = []
            for j in range(n):
                action.append(round(random.random()))
            for j in range(n):
                action.append(7.0 / n)
            num = sum(action[0:n])
            for j in range(n):
                if action[j] == 1:
                    action.append(10.0 / num)
                else:
                    action.append(0)
            content = ""
            for i in range(3 * n):
                content += str(action[i])
                if i != 3 * n - 1:
                    content += " "
            self.sock.send(content.encode('utf-8'))


    def allocateThresholdEquipartitionStrategy(self,n):
        pass

    def allocateEnumerateBestStrategy(self,n):
        agent = Agent(n=n, lr_critic=0, lr_actor=0, batch_size=0, target_update=0, gamma=0,
                      path=r"F:\Bupt\task-scheduling-and-resource-allocation-in-VR-video-service\nonlinear_programming\gnn_a2c_lm\GALModel.pkl")
        env = Env()
        map = agent.generate_map(n)
        while (True):
            data = self.sock.recv(1024)
            data = str(data, 'UTF-8')
            print(data)
            state = readState(data, n)
            minDelay = 9999
            minPolicy = None
            for action in map:
                policy = agent.action_to_policy(state, action)
                delay, _ = env.step(state, copy.deepcopy(policy), n)
                delay *= -1
                if delay < minDelay:
                    minDelay = delay
                    minPolicy = policy
            saveContent = ""
            minPolicy = minPolicy[0] + minPolicy[1] + minPolicy[2]
            for i in range(len(minPolicy)):
                saveContent += str(minPolicy[i])
                if i != len(minPolicy) - 1:
                    saveContent += " "
            self.sock.send(saveContent.encode('utf-8'))

def main():
    Allocator().allocateEnumerateBestStrategy(3)

if __name__ == "__main__":
    main()