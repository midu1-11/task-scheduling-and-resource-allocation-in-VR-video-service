package org.cloudbus.cloudsim.examples.mytest;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import sun.security.krb5.internal.PAData;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * @author howard
 * @version 1.0
 */
public class Utils {

    /**
     * 创建并返回任务列表
     */
    public static List<Cloudlet> createCloudlet(int userId, int cloudlets, int idShift) {
        //任务列表
        LinkedList<Cloudlet> list = new LinkedList<>();


        //long length = (long)(100*new Random().nextGaussian()+40000);
        /*length=任务cpu长度，fileSize=任务文件大小（目前没用），outputSize=任务输出文件大小（目前没用）
        pesNumber=任务所需cpu数量，目前就是1
        * */
        int min = Integer.parseInt(SysConfig.getInfo("cloudletLength").split("-")[0]);
        int max = Integer.parseInt(SysConfig.getInfo("cloudletLength").split("-")[1]);
        long length = (long) ((max - min) * Math.random() + min);
        long fileSize = Integer.parseInt(SysConfig.getInfo("cloudletFileSize"));//KB
        long outputSize = (long) (fileSize * Double.parseDouble(SysConfig.
                getInfo("cloudletOutputSize")));//KB
        int pesNumber = 1;

        //资源利用率，不知道干啥的
        UtilizationModel utilizationModel = new UtilizationModelFull();

        Cloudlet[] cloudlet = new Cloudlet[cloudlets];

        for (int i = 0; i < cloudlets; i++) {
            cloudlet[i] = new MyCloudlet(idShift + i, length, pesNumber, fileSize,
                    outputSize, utilizationModel, utilizationModel, utilizationModel, CloudSim.clock());
            // 设置任务的拥有者，这里认为是端设备
            cloudlet[i].setUserId(userId);
            list.add(cloudlet[i]);
        }

        return list;
    }

    /**
     * 创建并返回虚拟机列表
     */
    public static List<Vm> createVM(int userId, int vms, int idShift, int mips) {
        //虚拟机列表
        LinkedList<Vm> list = new LinkedList<Vm>();

        /*虚拟机参数
        size=磁盘大小，ram=内存，mips=cpu速度，bw=带宽，pesNumber=虚拟cpu数量（目前就是1）
        目前只调过mips
        * */
        long size = 10000; //image size (MB)
        int ram = 512; //vm memory (MB)
        //int mips = 250;
        long bw = 1000;
        int pesNumber = 1; //number of cpus
        String vmm = "Xen"; //VMM name

        Vm[] vm = new Vm[vms];

        for (int i = 0; i < vms; i++) {
            //CloudletSchedulerTimeShared指虚拟机cpu分时执行，同时多个任务会平分cpu的mips速度
            vm[i] = new Vm(idShift + i, userId, mips, pesNumber, ram, bw, size, vmm,
                    new CloudletSchedulerTimeShared());
            list.add(vm[i]);
        }

        return list;
    }

    /**
     * 创建并返回MEC设备
     */
    public static MyMec createMyMec(String name, int myCloudId) {

        // 主机列表
        List<Host> hostList = new ArrayList<>();

        // cpu列表
        List<Pe> peList = new ArrayList<>();

        int mips = Integer.parseInt(SysConfig.getInfo("mecMips"));

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

        // 创建MEC设备
        MyMec myMec = null;
        try {
            myMec = new MyMec(name, characteristics, new VmAllocationPolicySimple(hostList),
                    storageList, 0, myCloudId);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return myMec;
    }

    /**
     * 创建并返回Cloud设备
     */
    public static MyCloud createMyCloud(String name) {

        // 主机列表
        List<Host> hostList = new ArrayList<>();

        // cpu列表
        List<Pe> peList = new ArrayList<>();

        int mips = Integer.parseInt(SysConfig.getInfo("cloudMips"));

        // 这里一台主机只放了一个cpu
        peList.add(new Pe(0, new PeProvisionerSimple(mips)));


        /*主机参数
        hostId=主机id，ram=主机内存，storage=主机磁盘大小，bw=带宽，目前这几个参数没啥用，不超过Datacenter容量就行
        * */
        int hostId = 0;
        int ram = 16384; //host memory (MB)
        long storage = 1000000; //host storage
        int bw = Integer.parseInt(SysConfig.getInfo("cloudBandwidth"));

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

        // 创建MEC设备
        MyCloud myCloud = null;
        try {
            myCloud = new MyCloud(name, characteristics, new VmAllocationPolicySimple(hostList),
                    storageList, 0);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return myCloud;
    }


    /**
     * 创建并返回端设备
     */
    public static MyEdge createMyEdge(String name, int myMecId, int myCloudId, int bw) {

        MyEdge myEdge = null;
        try {
            myEdge = new MyEdge(name, myMecId, myCloudId, bw);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
        return myEdge;
    }

    /**
     * 打印任务结果
     *
     * @param list 任务列表
     */
    public static void printCloudletList(List<MyCloudlet> list) {
        int size = list.size();
        MyCloudlet mycloudlet;

        String indent = "    ";
        Log.printLine();
        Log.printLine("========== OUTPUT ==========");
        Log.printLine("Cloudlet ID" + indent + "STATUS" + indent +
                "Data center ID" + indent + "VM ID" + indent + indent + "执行时间" + indent +
                "开始执行时刻" + indent + "结束执行时刻" + indent + "MTP" + indent + "任务产生时刻"
                + indent + "任务解决时刻" + indent + "任务开销");

        DecimalFormat dft = new DecimalFormat("###.##");
        for (int i = 0; i < size; i++) {
            mycloudlet = list.get(i);
            Log.print(indent + mycloudlet.getCloudletId() + indent + indent);

            if (mycloudlet.getCloudletStatus() == Cloudlet.SUCCESS) {
                Log.print("SUCCESS");

                Log.printLine(indent + indent + mycloudlet.getResourceId() + indent + indent +
                        indent + mycloudlet.getVmId() + indent + indent + indent +
                        dft.format(mycloudlet.getActualCPUTime()) + indent + indent +
                        dft.format(mycloudlet.getExecStartTime()) + indent + indent +
                        indent + dft.format(mycloudlet.getFinishTime()) + indent + indent +
                        indent + dft.format(mycloudlet.getSolveTime() - mycloudlet.
                        getGenerateTime()) + indent + indent + indent + dft.format(mycloudlet.
                        getGenerateTime()) + indent + indent + indent + dft.format(mycloudlet.
                        getSolveTime()) + indent + indent + indent + dft.format(mycloudlet.
                        getCost()));
            }
        }
    }

    /**
     * 保存任务结果列表，格式为"id MTP"
     */
    public static void saveCloudletList(List<MyCloudlet> list, String tag) {
        list = bubbleSort(list);
        String path = "";
        if (tag.equals("baseline")) {
            path = "F:\\Bupt\\cloudsim\\modules\\cloudsim-examples\\src\\main\\java\\org\\" +
                    "cloudbus\\cloudsim\\examples\\mytest\\baseline.txt";
        } else if (tag.equals("algorithm1")) {
            path = "F:\\Bupt\\cloudsim\\modules\\cloudsim-examples\\src\\main\\java\\org\\" +
                    "cloudbus\\cloudsim\\examples\\mytest\\result.txt";
        }
        BufferedWriter bufferedWriter = null;
        try {
            bufferedWriter = new BufferedWriter(new FileWriter(path));
            for (MyCloudlet myCloudlet : list) {
                bufferedWriter.write(myCloudlet.getCloudletId() + " " + (myCloudlet.getSolveTime()
                        - myCloudlet.getGenerateTime()) + " " + myCloudlet.getCost());
                bufferedWriter.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                bufferedWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * 冒泡排序
     */
    private static List<MyCloudlet> bubbleSort(List<MyCloudlet> list) {
        for (int i = 1; i < list.size(); i++) {
            for (int j = 0; j < list.size() - i; j++) {
                if (list.get(j).getCloudletId() > list.get(j + 1).getCloudletId()) {
                    MyCloudlet myCloudlet = list.get(j);
                    list.set(j, list.get(j + 1));
                    list.set(j + 1, myCloudlet);
                }
            }
        }
        return list;
    }

    /**
     * 读取结果并绘制图表
     */
    public static void printResultChart(int tag) {
        String path = "";
        String chartName = "";
        if (tag == MyStatement.BASELINE_RESULT) {
            path = "F:\\Bupt\\cloudsim\\modules\\cloudsim-examples\\src\\main\\java\\org\\" +
                    "cloudbus\\cloudsim\\examples\\mytest\\baseline.txt";
            chartName = "云";
        } else if (tag == MyStatement.ALGORITHM_RESULT) {
            path = "F:\\Bupt\\cloudsim\\modules\\cloudsim-examples\\src\\main\\java\\org\\" +
                    "cloudbus\\cloudsim\\examples\\mytest\\result.txt";
            chartName = "云边结合";
        }

        BufferedReader bufferedReader = null;

        List<String> buffer = new ArrayList<>();

        try {
            String s;
            bufferedReader = new BufferedReader(new FileReader(path));
            while ((s = bufferedReader.readLine()) != null) {
                buffer.add(s);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                bufferedReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        // 创建数据
        double[] xData = new double[buffer.size()];
        double[] yData = new double[buffer.size()];

        for (int i = 0; i < buffer.size(); i++) {
            xData[i] = Double.parseDouble(buffer.get(i).split(" ")[0]);
            yData[i] = Double.parseDouble(buffer.get(i).split(" ")[1]);
        }

        // 创建图表
        org.knowm.xchart.XYChart chart = QuickChart.getChart(chartName, "x(次数)", "MTP(s)", "MTP(x)", xData, yData);

        // 进行绘制
        new SwingWrapper<XYChart>(chart).displayChart();
    }
}
