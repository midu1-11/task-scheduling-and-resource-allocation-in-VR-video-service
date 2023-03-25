package org.cloudbus.cloudsim.examples.mytest;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.CloudSimTags;
import org.cloudbus.cloudsim.core.SimEvent;
import org.cloudbus.cloudsim.lists.VmList;

import java.util.LinkedList;
import java.util.List;

/**
 * @author howard
 * @version 1.0
 */
public class MyEdge extends MyDatacenterBroker {

    private int cloudletId = 1;

    private int myMecId;
    private int myCloudId;

    private int bw;

    public MyEdge(String name, int myMecId, int myCloudId, int bw) throws Exception {
        super(name);
        this.myMecId = myMecId;
        this.myCloudId = myCloudId;
        this.bw = bw;
    }

    @Override
    public void processEvent(SimEvent ev) {

        switch (ev.getTag()) {
            case MyStatement.SUBMIT_ONE_CLOUDLET:
                List<Cloudlet> list = Utils.createCloudlet(getId(), 1, cloudletId++);
                super.submitCloudletList(list);
                super.bindCloudletToVm(list.get(0).getCloudletId(), 0);
                super.submitCloudlets();
                CloudSim.resumeSimulation();
                break;

            case CloudSimTags.VM_CREATE_ACK:
                break;

            default:
                super.processEvent(ev);
                break;
        }
    }

    @Override
    public void startEntity() {
        super.startEntity();

        List<Vm> list = Utils.createVM(getId(), 1, 0, Integer.parseInt(SysConfig.getInfo("mecMips")));

        //给MEC发创建vm的消息，这里没有考虑创建失败，所以需要保证MEC的Datacenter是可以架设这个vm的
        sendNow(myMecId, CloudSimTags.VM_CREATE_ACK, list.get(0));

        //由于不采用MyDatacenterBroker的vm默认分配方式，所以注释了MyDatacenterBroker类的195行，并做出如下配置
        submitVmList(list);
        getVmsCreatedList().addAll(list);
        getVmsToDatacentersMap().put(list.get(0).getId(), myMecId);


        list = Utils.createVM(getId(), 1, 1, Integer.parseInt(SysConfig.getInfo("cloudMips")));

        //给Cloud发创建vm的消息，这里没有考虑创建失败，所以需要保证MEC的Datacenter是可以架设这个vm的
        sendNow(myCloudId, CloudSimTags.VM_CREATE_ACK, list.get(0));

        //由于不采用MyDatacenterBroker的vm默认分配方式，所以注释了MyDatacenterBroker类的195行，并做出如下配置
        submitVmList(list);
        getVmsCreatedList().addAll(list);
        getVmsToDatacentersMap().put(list.get(0).getId(), myCloudId);

        for (int i = 1; i < Integer.parseInt(SysConfig.getInfo("cloudletNumber")); i++) {
            schedule(getId(), Integer.parseInt(SysConfig.getInfo("cloudletGeneratePeriod"))
                    * i, MyStatement.SUBMIT_ONE_CLOUDLET);
        }
    }

    @Override
    public void shutdownEntity() {
    }

    @Override
    public int getBw() {
        return bw;
    }

}
