package org.cloudbus.cloudsim.examples.simulation;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSimTags;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * @author howard
 * @version 1.0
 * 边缘服务器类继承数据中心类，默认只有一台主机，主机只有一个cpu，这个cpu可以架设多个虚拟机
 */
public class EdgeServer extends Datacenter {
    public EdgeServer(
            String name,
            DatacenterCharacteristics characteristics,
            VmAllocationPolicy vmAllocationPolicy,
            List<Storage> storageList,
            double schedulingInterval) throws Exception {
        super(name, characteristics, vmAllocationPolicy, storageList, schedulingInterval);
    }

    // 静态方法，用于创建一个边缘服务器
    public static EdgeServer createEdgeServer(String name) {

        // 主机列表
        List<Host> hostList = new ArrayList<>();

        // cpu列表
        List<Pe> peList = new ArrayList<>();

        int mips = 10000;

        // 这里一台主机只放了一个cpu
        peList.add(new Pe(0, new PeProvisionerSimple(mips)));


        /*主机参数
        hostId=主机id，ram=主机内存，storage=主机磁盘大小，bw=带宽，目前这几个参数没啥用，不超过Datacenter容量就行
        * */
        int hostId = 0;
        int ram = 16384; //host memory (MB)
        long storage = 1000000; //host storage
        int bw = 100000;

        //VmSchedulerTimeShared方式，即主机允许在一个cpu上架设多个vm虚拟机
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


        EdgeServer edgeServer = null;
        try {
            edgeServer = new EdgeServer(name, characteristics, new VmAllocationPolicySimple(hostList),
                    storageList, 0);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return edgeServer;
    }

    // 重写了Datacenter类的sendCloudletReturn方法，考虑了任务返回客户端的通信延迟，即边缘服务器到客户端的延迟
    @Override
    public void sendCloudletReturn(Cloudlet cl) {
        int userId = cl.getUserId();
        int vmId = cl.getVmId();
        Host host = getVmAllocationPolicy().getHost(vmId, userId);
        Vm vm = host.getVm(vmId, userId);
        double delay = cl.getCloudletOutputSize() / (double) vm.getBw();
        send(cl.getUserId(), delay, CloudSimTags.CLOUDLET_RETURN, cl);
    }

    // 重写父类的updateCloudletProcessing方法，父类这个方法的主要作用是对于当前数据中心的所有主机的所有
    // 虚拟机上的所有任务，更新任务已完成的cpu长度，将已完成的任务返回并从虚拟机中清除，预估所有任务的完成时间，
    // 求所有任务完成时间中的最小时间并做一个事件调度，在最小时间后再次调用updateCloudletProcessing方法进行
    // 更新。
    @Override
    protected void updateCloudletProcessing() {
        // 更新所有任务进度并做出一个无效事件调度（因为没考虑虚拟机间计算资源转移）
        super.updateCloudletProcessing();

        // 边缘服务器唯一一台主机
        Host host = getVmAllocationPolicy().getHostList().get(0);
        // 这台主机的所有虚拟机的现有cpu速度
        Map<String, List<Double>> map = host.getVmScheduler().getMipsMap();

        // 统计未执行完任务的虚拟机数量
        int count = 0;
        for (Vm vm : getVmList()) {
            if (vm.getCloudletScheduler().getCloudletExecList().size() != 0) {
                count++;
            }
        }

        // 将本次唯一一个正好执行完任务的虚拟机的cpu速度，平均分给所有未执行完任务的虚拟机
        for (Vm vm : getVmList()) {
            if (vm.getMips() != 0 && vm.getCloudletScheduler().getCloudletExecList().size() == 0) {
                for (Vm vm1 : getVmList()) {
                    if (vm1.getCloudletScheduler().getCloudletExecList().size() != 0) {
                        double a = map.get(vm1.getUid()).get(0) + map.get(vm.getUid()).get(0) / count;
                        ((MyCloudletScheduler) (vm1.getCloudletScheduler())).setMips(a);
                    }
                }
                vm.setMips(0);
            }
        }
        // 再次更新所有任务进度（时间没变所以相当于没更新），并做出正确的的事件调度（考虑了虚拟机间计算资源转移）
        super.updateCloudletProcessing();
    }
}
