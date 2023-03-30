package org.cloudbus.cloudsim.examples.simulation;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.UtilizationModel;
import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.examples.mytest.SysConfig;

/**
 * @author howard
 * @version 1.0
 * MyCloudlet类继承Cloudlet类
 */
public class MyCloudlet extends Cloudlet {
    private static int cloudletNum = 1;
    private double pt;
    // 记录任务的产生时间以及客户端收到任务结果的时间
    private double startTime,endTime;

    public MyCloudlet(
            final int cloudletId,
            final long cloudletLength,
            final int pesNumber,
            final long cloudletFileSize,
            final long cloudletOutputSize,
            final UtilizationModel utilizationModelCpu,
            final UtilizationModel utilizationModelRam,
            final UtilizationModel utilizationModelBw) {
        super(cloudletId, cloudletLength, pesNumber, cloudletFileSize, cloudletOutputSize,
                utilizationModelCpu, utilizationModelRam, utilizationModelBw);
        startTime = CloudSim.clock();
    }

    // 静态方法，用于产生一个任务
    public static MyCloudlet createOneCloudlet(int userId) {
        // 任务的cpu长度
        long length = (long) (1360000 * Math.random() + 400000);

        // 任务大小，在VR视频业务中认为任务的数据量很小
        long fileSize = 0;//KB

        // 任务结果大小
        long outputSize = (long) (86000 * Math.random() + 14000);//KB
        int pesNumber = 1;

        //资源利用率，不知道干啥的
        UtilizationModel utilizationModel = new UtilizationModelFull();

        MyCloudlet cloudlet = new MyCloudlet(cloudletNum++, length, pesNumber, fileSize,
                outputSize, utilizationModel, utilizationModel, utilizationModel);

        // 设置任务的拥有者为相应的客户端
        cloudlet.setUserId(userId);
        return cloudlet;
    }

    public double getPt() {
        return pt;
    }

    public void setPt(double pt) {
        this.pt = pt;
    }

    public double getStartTime() {
        return startTime;
    }

    public void setStartTime(double startTime) {
        this.startTime = startTime;
    }

    public double getEndTime() {
        return endTime;
    }

    public void setEndTime(double endTime) {
        this.endTime = endTime;
    }
}
