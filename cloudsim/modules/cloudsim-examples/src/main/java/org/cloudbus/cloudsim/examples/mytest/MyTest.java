package org.cloudbus.cloudsim.examples.mytest;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.NetworkTopology;
import org.cloudbus.cloudsim.core.CloudSim;

import javax.rmi.CORBA.Util;
import java.util.Calendar;
import java.util.List;

/**
 * @author howard
 * @version 1.0
 */
public class MyTest {
    public static void main(String[] args) {
        int num_user = 1;   // number of grid users
        int cloudletIdShift = 0;
        Calendar calendar = Calendar.getInstance();
        boolean trace_flag = false;  // mean trace events

        // Initialize the CloudSim library
        CloudSim.init(num_user, calendar, trace_flag);

        //MyCloud myCloud = Utils.createMyMec("边缘服务器");

        //这里要注意myMec对象创建完之后会被默认保存到实体列表中，在底层myEdge的父类Datacenter有着固定的
        //vm分配策略，即先把所有vm发给实体列表中的第一个Datacenter，再把没有创建成功的vm发给实体列表中的
        // 第二个Datacenter，以此类推，如果我们想要控制vm的创建，就需要重写Datacenter的某些方法
        MyCloud myCloud = Utils.createMyCloud("云服务器");
        MyMec myMec = Utils.createMyMec("边缘服务器", myCloud.getId());

        MyEdge myEdge = Utils.createMyEdge("端设备", myMec.getId(), myCloud.getId(),
                Integer.parseInt(SysConfig.getInfo("edgeBandwidth")));


        NetworkTopology.buildNetworkTopology("F:\\Bupt\\cloudsim\\modules\\" +
                "cloudsim-examples\\src\\main\\java\\org\\cloudbus\\cloudsim\\examples\\" +
                "mytest\\1.brite");

        int briteNode = 0;
        NetworkTopology.mapNode(myEdge.getId(), briteNode);

        briteNode = 1;
        NetworkTopology.mapNode(myMec.getId(), briteNode);

        briteNode = 2;
        NetworkTopology.mapNode(myCloud.getId(), briteNode);


        CloudSim.startSimulation();

        // Final step: Print results when simulation is over
        List<MyCloudlet> newList = myEdge.getCloudletReceivedList();
        //newList.addAll(globalBroker.getBroker().getCloudletReceivedList());

        CloudSim.stopSimulation();

        Utils.printCloudletList(newList);
        Utils.saveCloudletList(newList, SysConfig.getInfo("algorithm"));

        Log.printLine("CloudSimExample8 finished!");
    }
}
