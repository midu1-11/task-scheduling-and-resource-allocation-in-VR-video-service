package org.cloudbus.cloudsim.examples.simulation;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.CloudSimTags;
import org.cloudbus.cloudsim.core.SimEntity;
import org.cloudbus.cloudsim.core.SimEvent;
import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

/**
 * @author howard
 * @version 1.0
 * 边缘中心控制器类，继承了DatacenterBroker类，主要作用是接收所有客户端的任务，并做出任务调度和资源分配
 */
public class Controller extends DatacenterBroker {

    // 和python端通过socket通信
    private ServerSocket serverSocket = new ServerSocket(9999);
    private Socket socket;

    // 保持连接的客户端列表
    private List<Client> clientList = new ArrayList<>();

    // 保存每个时隙的所有客户端任务
    private List<Cloudlet> myCloudletList = new ArrayList<>();
    private static int vmIdShift = 0;

    // 边缘服务器
    private SimEntity EdgeServer;
    // 云服务器
    private SimEntity CloudServer;

    private List<Double> delayList = new ArrayList<>();

    // 记录了每个时隙所有客户端的总延迟
    private List<Double> delayRecordList = new ArrayList<>();

    public Controller(String name, SimEntity EdgeServer, SimEntity CloudServer)
            throws Exception {
        super(name);
        this.EdgeServer = EdgeServer;
        this.CloudServer = CloudServer;
        socket = serverSocket.accept();
    }

    @Override
    public void processEvent(SimEvent ev) {
        switch (ev.getTag()) {
            // 某客户端发起连接
            case MyTags.CONNECT_ONE_CLIENT:
                clientList.add((Client) ev.getData());
                break;

            // 某客户端断开连接
            case MyTags.DISCONNECT_ONE_CLIENT:
                clientList.remove((Client) ev.getData());
                break;

            // 收到某一时隙的所有客户端的任务后，首先进行资源分配，然后进行任务调度
            case MyTags.SUBMIT_ONE_CLOUDLET:
                myCloudletList.add((MyCloudlet) ev.getData());
                if (myCloudletList.size() == clientList.size()) {
                    processAllocate();
                    processCloudletSubmit();
                }
                break;

            // 保存某一时隙的所有客户端的延迟之和
            case MyTags.SUBMIT_ONE_DELAY_RECORD:
                delayList.add((Double) ev.getData());
                Double delaySum = 0.0;
                if (delayList.size() == clientList.size()) {
                    for (Double delay : delayList) {
                        delaySum += delay;
                    }
                    delayRecordList.add(delaySum);
                    delayList.clear();
                }
                break;

            // 其他事件走父类
            default:
                super.processEvent(ev);
        }
    }

    // 将任务信息、网络信息、云服务器信息转化为特定字符串格式
    private String processState() {
        StringBuffer stringBuffer = new StringBuffer();
        double pt = Math.abs(new Random().nextGaussian() + 0.5);
        for (Cloudlet cloudlet : myCloudletList) {
            ((MyCloudlet) cloudlet).setPt(pt);
            stringBuffer.append(cloudlet.getCloudletOutputSize() / 1000.0);
            stringBuffer.append(" ");
            stringBuffer.append(cloudlet.getCloudletLength() / 1000.0);
            stringBuffer.append(" ");
        }
        stringBuffer.append(pt);
        stringBuffer.append(" ");
        stringBuffer.append(5);
        return stringBuffer.toString();
    }

    // 进行资源分配
    private void processAllocate() {
        OutputStream outputStream = null;
        InputStream inputStream = null;
        byte[] buf = new byte[1024];
        int readLen = 0;
        try {
            // 通过socket的TCP通信拿到python端返回的决策结果
            outputStream = socket.getOutputStream();
            String state = processState();
            outputStream.write(state.getBytes());
            inputStream = socket.getInputStream();
            readLen = inputStream.read(buf);
            String result = new String(buf, 0, readLen);
            System.out.println(CloudSim.clock() + ": " + result);

            // 清空上一时隙创建的所有虚拟机
            clearAllVms();

            // 根据上面返回的决策结果创建虚拟机
            createVms(result + " " + state.split(" ")[2 * clientNum()] + " "
                    + state.split(" ")[2 * clientNum() + 1]);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // 提交当前时隙的所有客户端任务
    private void processCloudletSubmit() {
        super.submitCloudletList(myCloudletList);
        for (int i = 0; i < myCloudletList.size(); i++) {
            // 将任务绑定给相应的虚拟机处理
            super.bindCloudletToVm(myCloudletList.get(i).getCloudletId(), i + 1);
        }
        super.submitCloudlets();

        // 清空客户端任务列表
        myCloudletList.clear();
        CloudSim.resumeSimulation();
    }

    // 清空创建的所有虚拟机
    private void clearAllVms() {
        List<Vm> vmList = getVmList();
        for (Vm vm : vmList) {
            sendNow(getVmsToDatacentersMap().get(vm.getId()), CloudSimTags.VM_DESTROY, vm);
        }
        getVmList().clear();
        getVmsCreatedList().clear();
        getVmsToDatacentersMap().clear();
    }

    // 根据资源分配和任务调度的结果，为当前时隙每个任务创建一个虚拟机
    private void createVms(String result) {
        vmIdShift = 0;
        String[] allocation = result.split(" ");

        // 任务调度结果
        int[] decision = new int[clientNum()];
        // 带宽分配结果
        int[] bandwidthAlloaction = new int[clientNum()];
        // 边缘服务器cpu速度分配结果
        int[] cpuSpeedAlloaction = new int[clientNum()];

        double pt;
        int sc;

        for (int i = 0; i < clientNum(); i++) {
            decision[i] = Integer.parseInt(allocation[i]);
        }
        for (int i = 0; i < clientNum(); i++) {
            bandwidthAlloaction[i] = (int) (Double.parseDouble(allocation[i + clientNum()]) * 1000);
        }
        for (int i = 0; i < clientNum(); i++) {
            cpuSpeedAlloaction[i] = (int) (Double.parseDouble(allocation[i + clientNum() * 2]) * 1000);
        }
        pt = Double.parseDouble(allocation[clientNum() * 3]);
        sc = (int) (Double.parseDouble(allocation[clientNum() * 3 + 1]) * 1000);

        for (int i = 0; i < clientNum(); i++) {
            // 云端创建虚拟机
            if (decision[i] == 0) {
                List<Vm> vmList = createOneVm(clientList.get(i).getId(), sc, bandwidthAlloaction[i]);
                //给云服务器发创建vm的消息，这里没有考虑创建失败，所以需要保证云服务器现有资源是可以架设这个vm的
                sendNow(CloudServer.getId(), CloudSimTags.VM_CREATE_ACK, vmList.get(0));

                //由于不采用DatacenterBroker的vm默认分配方式，所以注释了DatacenterBroker类的195行，并做出如下配置
                submitVmList(vmList);
                getVmsCreatedList().addAll(vmList);
                getVmsToDatacentersMap().put(vmList.get(0).getId(), CloudServer.getId());
            }
            // 边缘创建虚拟机
            else if (decision[i] == 1) {
                List<Vm> vmList = createOneVm(clientList.get(i).getId(), cpuSpeedAlloaction[i], bandwidthAlloaction[i]);
                //给边缘服务器发创建vm的消息，这里没有考虑创建失败，所以需要保证边缘服务器是可以架设这个vm的
                sendNow(EdgeServer.getId(), CloudSimTags.VM_CREATE_ACK, vmList.get(0));

                //由于不采用DatacenterBroker的vm默认分配方式，所以注释了DatacenterBroker类的195行，并做出如下配置
                submitVmList(vmList);
                getVmsCreatedList().addAll(vmList);
                getVmsToDatacentersMap().put(vmList.get(0).getId(), EdgeServer.getId());
            }
        }
    }

    // 静态方法，用于创建一个虚拟机
    private static List<Vm> createOneVm(int userId, int mips, long bw) {
        //虚拟机列表
        LinkedList<Vm> list = new LinkedList<>();

        /*虚拟机参数
        size=磁盘大小，ram=内存，mips=cpu速度，bw=带宽，pesNumber=虚拟cpu数量（目前就是1）
        目前只调过mips
        * */
        long size = 10000; //image size (MB)
        int ram = 512; //vm memory (MB)
        int pesNumber = 1; //number of cpus
        String vmm = "Xen"; //VMM name

        // 虚拟机调度器是MyCloudletScheduler对象，此调度器会在任意时隙当虚拟机上唯一的任务执行完后，将虚拟机
        // 剩余的cpu速度平分给还在执行任务的所有其他虚拟机，原来这里的调度器是CloudletSchedulerTimeShared，
        // 即虚拟机cpu分时执行，同时多个任务会平分cpu的mips速度
        Vm vm = new Vm(++vmIdShift, userId, mips, pesNumber, ram, bw, size, vmm,
                new MyCloudletScheduler((double) mips));
        list.add(vm);

        return list;
    }

    // 当前连接客户端数
    private int clientNum() {
        return clientList.size();
    }

    // 打印所有时隙平均延迟
    @Override
    public void shutdownEntity() {
        super.shutdownEntity();
        int delayNum = delayRecordList.size();
        double delaySum = 0;
        for (Double delay : delayRecordList) {
            delaySum += delay;
        }
        System.out.println("平均延迟：" + delaySum / delayNum);
    }
}
