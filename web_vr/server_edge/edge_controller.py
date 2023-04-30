from graph_neural_network_advantage_actor_critic_lagrange_multiplier import *

class Controller:
    def __init__(self):
        # 初始化一个智能体对象，这里模型用的是之前训练好的模型
        self.agent = Agent(n=6, lr_critic=0, lr_actor=0, batch_size=0, target_update=0, gamma=0,
                      path=r"F:\Bupt\task-scheduling-and-resource-allocation-in-VR-video-service\web_vr\server_edge\GALModel.pkl")

    def max(self, a, b):
        if a > b:
            return a
        else:
            return b

    # 返回任务调度结果
    def get_decision_list(self):
        b_d = []
        for i in range(6):
            b_d.append([0.086 * random.random() + 0.014, 1.36 * random.random() + 0.4])
        pt_sc = [self.max(0, random.gauss(0.04, 0.02)), 0.007]

        state = (b_d, pt_sc)
        action, _ = self.agent.interacte(state)

        decision_list = []
        for a in action:
            if a==1:
                decision_list.append(True)
            else:
                decision_list.append(False)

        return decision_list
