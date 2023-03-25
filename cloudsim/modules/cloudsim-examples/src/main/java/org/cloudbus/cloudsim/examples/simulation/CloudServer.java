package org.cloudbus.cloudsim.examples.simulation;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSimTags;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * @author howard
 * @version 1.0
 * 云服务器继承了数据中心类，默认只有一台主机，主机只有一个cpu，这个cpu可以架设多个虚拟机
 */
public class CloudServer extends Datacenter {

    public CloudServer(
            String name,
            DatacenterCharacteristics characteristics,
            VmAllocationPolicy vmAllocationPolicy,
            List<Storage> storageList,
            double schedulingInterval) throws Exception {
        super(name, characteristics, vmAllocationPolicy, storageList, schedulingInterval);
    }

    // 静态方法，用于创建一个云服务器对象
    public static CloudServer createCloudServer(String name) {

        // 主机列表
        List<Host> hostList = new ArrayList<>();

        // cpu列表
        List<Pe> peList = new ArrayList<>();

        // 云服务器的cpu速度近乎无限
        int mips = 999999;

        // 这里一台主机只放了一个cpu
        peList.add(new Pe(0, new PeProvisionerSimple(mips)));


        /*主机参数
        hostId=主机id，ram=主机内存，storage=主机磁盘大小，bw=带宽，目前这几个参数没啥用，不超过Datacenter容量就行
        * */
        int hostId = 0;
        int ram = 16384; //host memory (MB)
        long storage = 1000000; //host storage
        int bw = 100000;

        // VmSchedulerTimeShared方式，即主机允许在一个cpu上架设多个vm虚拟机
        hostList.add(
                new Host(
                        hostId,
                        new RamProvisionerSimple(ram),
                        new BwProvisionerSimple(bw),
                        storage,
                        peList,
                        new VmSchedulerTimeShared(peList)
                )
        );

        /*arch=系统架构，os=操作系统，time_zone=时区，cost=cpu每秒花费，costPerMem=单位内存花费
        costPerStorage=单位磁盘大小花费，costPerBw=单位带宽花费，这些参数目前都没调过
        * */
        String arch = "x86";      // system architecture
        String os = "Linux";          // operating system
        String vmm = "Xen";
        double time_zone = 10.0;         // time zone this resource located
        double cost = 3.0;              // the cost of using processing in this resource
        double costPerMem = 0.05;        // the cost of using memory in this resource
        double costPerStorage = 0.1;    // the cost of using storage in this resource
        double costPerBw = 0.1;            // the cost of using bw in this resource
        LinkedList<Storage> storageList = new LinkedList<Storage>();    //we are not adding SAN devices by now

        DatacenterCharacteristics characteristics = new DatacenterCharacteristics(
                arch, os, vmm, hostList, time_zone, cost, costPerMem, costPerStorage, costPerBw);

        CloudServer cloudServer = null;
        try {
            cloudServer = new CloudServer(name, characteristics, new VmAllocationPolicySimple(hostList),
                    storageList, 0);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return cloudServer;
    }

    // 重写了Datacenter类的sendCloudletReturn方法，考虑了任务返回客户端的通信延迟，包括
    // 云服务器到边缘服务器的延迟以及边缘服务器到客户端的延迟
    @Override
    public void sendCloudletReturn(Cloudlet cl) {
        int userId = cl.getUserId();
        int vmId = cl.getVmId();
        Host host = getVmAllocationPolicy().getHost(vmId, userId);
        Vm vm = host.getVm(vmId, userId);
        double delay = cl.getCloudletOutputSize() / (double) vm.getBw() + ((MyCloudlet) cl).getPt();
        send(cl.getUserId(), delay, CloudSimTags.CLOUDLET_RETURN, cl);
    }
}
