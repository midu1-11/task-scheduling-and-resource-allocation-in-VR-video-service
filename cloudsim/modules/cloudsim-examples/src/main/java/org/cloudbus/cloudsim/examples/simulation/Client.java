package org.cloudbus.cloudsim.examples.simulation;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.CloudSimTags;
import org.cloudbus.cloudsim.core.SimEntity;
import org.cloudbus.cloudsim.core.SimEvent;


/**
 * @author howard
 * @version 1.0
 * 客户端，也即用户，可以发起任务请求
 */
public class Client extends SimEntity {

    // 对应的中心控制器
    private SimEntity controller;

    public Client(String name, SimEntity controller) {
        super(name);
        this.controller = controller;
    }

    // 连接中心控制器
    private void processConnect() {
        sendNow(controller.getId(), MyTags.CONNECT_ONE_CLIENT, this);
    }

    // 断开中心控制器的连接
    private void processDisconnect() {
        sendNow(controller.getId(), MyTags.DISCONNECT_ONE_CLIENT, this);
    }

    @Override
    public void startEntity() {
        processConnect();
        // 按照给定频率产生任务
        for (int i = 0; i < 1000; i++) {
            schedule(getId(), 10000 * i, MyTags.SUBMIT_ONE_CLOUDLET);
        }
    }

    @Override
    public void processEvent(SimEvent ev) {
        switch (ev.getTag()) {

            // 产生一个任务并提交给中心控制器
            case MyTags.SUBMIT_ONE_CLOUDLET:
                MyCloudlet cloudlet = MyCloudlet.createOneCloudlet(getId());
                sendNow(controller.getId(), MyTags.SUBMIT_ONE_CLOUDLET, cloudlet);
                break;

            // 收到任务结果，打印信息并将延迟反馈给中心控制器
            case CloudSimTags.CLOUDLET_RETURN:
                MyCloudlet returnedCloudlet = (MyCloudlet) ev.getData();
                returnedCloudlet.setEndTime(CloudSim.clock());
                System.out.println(CloudSim.clock() + ":客户端#" + getId() + "收到任务#" + returnedCloudlet.getCloudletId());
                sendNow(controller.getId(), MyTags.SUBMIT_ONE_DELAY_RECORD,
                        returnedCloudlet.getEndTime() - returnedCloudlet.getStartTime());
                break;

            default:
                break;
        }
    }

    @Override
    public void shutdownEntity() {}
}
