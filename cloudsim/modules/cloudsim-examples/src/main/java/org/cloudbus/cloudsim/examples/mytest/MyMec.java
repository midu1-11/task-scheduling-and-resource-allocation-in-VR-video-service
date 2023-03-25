package org.cloudbus.cloudsim.examples.mytest;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.CloudSimTags;
import org.cloudbus.cloudsim.core.SimEvent;

import java.util.List;
import java.util.Random;

/**
 * @author howard
 * @version 1.0
 */
public class MyMec extends Datacenter {

    private int myCloudId;

    public MyMec(String name, DatacenterCharacteristics characteristics,
                 VmAllocationPolicy vmAllocationPolicy, List<Storage> storageList,
                 double schedulingInterval, int myCloudId) throws Exception {
        super(name, characteristics, vmAllocationPolicy, storageList, schedulingInterval);
        this.myCloudId = myCloudId;
    }

    @Override
    public void processEvent(SimEvent ev) {

        switch (ev.getTag()) {
            case MyStatement.CHANGE_HOST_BW:
                int mean = Integer.parseInt(SysConfig.getInfo("mecBandwidth").split("-")[0]);
                int variance = Integer.parseInt(SysConfig.getInfo("mecBandwidth").split("-")[1]);
                super.getCharacteristics().getHostList().get(0).getBwProvisioner().
                        setBw((long) (new Random().nextGaussian() * Math.sqrt(variance) + mean));
                break;

            case CloudSimTags.CLOUDLET_SUBMIT:
                long bw = super.getCharacteristics().getHostList().get(0).getBwProvisioner().getBw();

                if (bw > -1) { //60000
                    System.out.println(CloudSim.clock() + ":边缘服务器处理 " + ((Cloudlet) ev.getData()).
                            getCloudletId() + "号任务");
                    super.processEvent(ev);
                } else {
                    System.out.println(CloudSim.clock() + ":边缘服务器将 " + ((Cloudlet) ev.getData()).
                            getCloudletId() + "号任务发往云服务器");
                    //将任务的VmId改成云服务器上的vm的id
                    ((Cloudlet) ev.getData()).setVmId(1);
                    //边缘服务器到云服务器的延迟=发送延迟+传输延迟，这里的sendingDelay是发送延迟，等于
                    //任务大小除以边缘服务器唯一一台主机的带宽，这里通过让带宽随时间变化模拟网络变化
                    double sendingDelay = ((Cloudlet) ev.getData()).getCloudletFileSize() * 1024 * 8
                            / (double) bw;
                    send(myCloudId, sendingDelay, CloudSimTags.CLOUDLET_SUBMIT, (Cloudlet) ev.getData());
                }
                break;

            default:
                super.processEvent(ev);
                break;
        }
    }

    @Override
    public void startEntity() {
        super.startEntity();

        for (int i = 1; i < Integer.parseInt(SysConfig.getInfo("cloudletNumber")); i++) {
            schedule(getId(), Integer.parseInt(SysConfig.getInfo("cloudletGeneratePeriod"))
                    * i, MyStatement.CHANGE_HOST_BW);
        }
    }

    @Override
    public void sendCloudletReturn(Cloudlet cl) {
        //边缘服务器执行完任务返回时，要计算发送延迟,注意这里文件大小是getCloudletOutputSize()
        MyCloudlet myCloudlet = (MyCloudlet) cl;
        long bw = super.getCharacteristics().getHostList().get(0).getBwProvisioner().getBw();
        double sendingDelay = myCloudlet.getCloudletOutputSize() * 1024 * 8 / (double) bw;
        myCloudlet.setExcutedByCloud(false);
        send(myCloudlet.getUserId(), sendingDelay, CloudSimTags.CLOUDLET_RETURN, myCloudlet);
    }
}
