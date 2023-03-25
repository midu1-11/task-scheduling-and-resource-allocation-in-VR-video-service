package org.cloudbus.cloudsim.examples.mytest;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.UtilizationModel;

import java.io.Serializable;
import java.util.List;

/**
 * @author howard
 * @version 1.0
 */
public class MyCloudlet extends Cloudlet {

    private static final long serialVersionUID = 1L;

    //任务在边缘产生的时刻
    private Double generateTime;

    //边缘收到解决完的任务的时刻
    private Double solveTime;

    //记录任务是由边缘服务器执行的还是云服务器执行的
    private boolean isExcutedByCloud = false;

    //任务总的开销
    private double cost;

    public MyCloudlet(
            final int cloudletId,
            final long cloudletLength,
            final int pesNumber,
            final long cloudletFileSize,
            final long cloudletOutputSize,
            final UtilizationModel utilizationModelCpu,
            final UtilizationModel utilizationModelRam,
            final UtilizationModel utilizationModelBw,Double generateTime) {
        super(cloudletId,cloudletLength,pesNumber,cloudletFileSize,cloudletOutputSize,
                utilizationModelCpu,utilizationModelRam,utilizationModelBw);
        this.generateTime = generateTime;
    }

    public MyCloudlet(
            final int cloudletId,
            final long cloudletLength,
            final int pesNumber,
            final long cloudletFileSize,
            final long cloudletOutputSize,
            final UtilizationModel utilizationModelCpu,
            final UtilizationModel utilizationModelRam,
            final UtilizationModel utilizationModelBw,
            final boolean record,
            final List<String> fileList) {
        super(cloudletId,cloudletLength,pesNumber,cloudletFileSize,cloudletOutputSize,
                utilizationModelCpu,utilizationModelRam,utilizationModelBw,record,fileList);
    }

    public MyCloudlet(
            final int cloudletId,
            final long cloudletLength,
            final int pesNumber,
            final long cloudletFileSize,
            final long cloudletOutputSize,
            final UtilizationModel utilizationModelCpu,
            final UtilizationModel utilizationModelRam,
            final UtilizationModel utilizationModelBw,
            final List<String> fileList) {
        super(cloudletId,cloudletLength,pesNumber,cloudletFileSize,cloudletOutputSize,
                utilizationModelCpu,utilizationModelRam,utilizationModelBw,fileList);
    }

    public MyCloudlet(
            final int cloudletId,
            final long cloudletLength,
            final int pesNumber,
            final long cloudletFileSize,
            final long cloudletOutputSize,
            final UtilizationModel utilizationModelCpu,
            final UtilizationModel utilizationModelRam,
            final UtilizationModel utilizationModelBw,
            final boolean record) {
        super(cloudletId,cloudletLength,pesNumber,cloudletFileSize,cloudletOutputSize,
                utilizationModelCpu,utilizationModelRam,utilizationModelBw,record);
    }

    public Double getSolveTime() {
        return solveTime;
    }

    public void setSolveTime(Double solveTime) {
        this.solveTime = solveTime;
    }

    public Double getGenerateTime() {
        return generateTime;
    }

    public void setGenerateTime(Double generateTime) {
        this.generateTime = generateTime;
    }

    public boolean isExcutedByCloud() {
        return isExcutedByCloud;
    }

    public void setExcutedByCloud(boolean excutedByCloud) {
        isExcutedByCloud = excutedByCloud;
    }

    public double getCost() {
        return cost;
    }

    public void setCost(double cost) {
        this.cost = cost;
    }
}
