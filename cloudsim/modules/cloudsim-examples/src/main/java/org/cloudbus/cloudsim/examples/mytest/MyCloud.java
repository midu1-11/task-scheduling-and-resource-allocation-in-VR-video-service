package org.cloudbus.cloudsim.examples.mytest;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSimTags;

import java.util.List;

/**
 * @author howard
 * @version 1.0
 */
public class MyCloud extends Datacenter {

    public MyCloud(String name, DatacenterCharacteristics characteristics,
                   VmAllocationPolicy vmAllocationPolicy, List<Storage> storageList,
                   double schedulingInterval) throws Exception {
        super(name, characteristics, vmAllocationPolicy, storageList, schedulingInterval);
    }

    @Override
    public void sendCloudletReturn(Cloudlet cl) {
        //云服务器执行完任务返回时，要计算发送延迟,注意这里文件大小是getCloudletOutputSize()
        MyCloudlet myCloudlet = (MyCloudlet)cl;
        long bw = super.getCharacteristics().getHostList().get(0).getBwProvisioner().getBw();
        double sendingDelay = myCloudlet.getCloudletOutputSize() * 1024 * 8 / (double) bw;
        myCloudlet.setExcutedByCloud(true);
        send(myCloudlet.getUserId(), sendingDelay, CloudSimTags.CLOUDLET_RETURN, myCloudlet);
    }

}
