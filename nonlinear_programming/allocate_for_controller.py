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
    def __init__(self,w,s):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(('localhost',9999))
        self.w = w
        self.s = s

    def allocateGALStrategy(self,n):
        agent = Agent(n=n, lr_critic=0, lr_actor=0, batch_size=0, target_update=0, gamma=0,
                      path=r"F:\Bupt\task-scheduling-and-resource-allocation-in-VR-video-service\nonlinear_programming\gnn_a2c_lm\GALModel.pkl")
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
            a = saveContent.encode('utf-8')
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
                action.append(random.random() * 4 + 0.1)
            for j in range(n):
                if action[j] == 1:
                    action.append(random.random() * 4 + 0.1)
                else:
                    action.append(0)
            sum_w = sum(action[n:2 * n])
            sum_b = sum(action[2 * n:3 * n])
            for j in range(n):
                action[n + j] = action[n + j] / sum_w * self.w
                if sum_b != 0:
                    action[2 * n + j] = action[2 * n + j] / sum_b * self.s

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
                action.append(self.w / n)
            num = sum(action[0:n])
            for j in range(n):
                if action[j] == 1:
                    action.append(self.s / num)
                else:
                    action.append(0)
            content = ""
            for i in range(3 * n):
                content += str(action[i])
                if i != 3 * n - 1:
                    content += " "
            self.sock.send(content.encode('utf-8'))

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

    def allocateRandomLagrangeMultiplierStrategy(self, n):
        agent = Agent(n=n, lr_critic=0, lr_actor=0, batch_size=0, target_update=0, gamma=0,
                      path=r"F:\Bupt\task-scheduling-and-resource-allocation-in-VR-video-service\nonlinear_programming\gnn_a2c_lm\GALModel.pkl")
        env = Env()
        map = agent.generate_map(n)
        while (True):
            data = self.sock.recv(1024)
            data = str(data, 'UTF-8')
            print(data)
            state = readState(data, n)
            action = random.sample(map, 1)[0]
            policy = agent.action_to_policy(state, action)
            saveContent = ""
            savePolicy = policy[0] + policy[1] + policy[2]
            for i in range(len(savePolicy)):
                saveContent += str(savePolicy[i])
                if i != len(savePolicy) - 1:
                    saveContent += " "
            self.sock.send(saveContent.encode('utf-8'))

    def allocateThresholdProportionalStrategy(self,n):
        while (True):
            data = self.sock.recv(1024)
            data = str(data, 'UTF-8')
            print(data)
            state = readState(data, n)
            b_d = state[0]
            action = []
            for j in range(n):
                if b_d[j][1] > 1.6:
                    action.append(0)
                else:
                    action.append(1)
            sum_b = 0
            sum_d = 0
            for j in range(n):
                sum_b += b_d[j][0]
                if action[j] == 1:
                    sum_d += b_d[j][1]
            for j in range(n):
                action.append(b_d[j][0] / sum_b * self.w)
            for j in range(n):
                if action[j] == 1:
                    action.append(b_d[j][1] / sum_d * self.s)
                else:
                    action.append(0)

            saveContent = ""
            for i in range(3 * n):
                saveContent += str(action[i])
                if i != 3 * n - 1:
                    saveContent += " "

            self.sock.send(saveContent.encode('utf-8'))


def main():
    n = 8
    # Allocator(100.0,200.0).allocateEnumerateBestStrategy(n)
    Allocator(100.0,200.0).allocateGALStrategy(n)
    # Allocator(100.0,200.0).allocateThresholdProportionalStrategy(n)
    # Allocator(100.0,200.0).allocateRandomEquipartitionStrategy(n)
    # Allocator(100.0,200.0).allocateRandomStrategy(n)

if __name__ == "__main__":
    main()