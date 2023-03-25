package org.cloudbus.cloudsim.examples.mytest;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.DatacenterBroker;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEntity;
import org.cloudbus.cloudsim.core.SimEvent;

import java.util.List;

/**
 * @author howard
 * @version 1.0
 */
public class MecBroker extends SimEntity {

    public final static int CLOUDLET_SUBMIT_TO_MEC = 1;

    public MecBroker(String name) {
        super(name);
    }

    @Override
    public void processEvent(SimEvent ev) {
        switch (ev.getTag()) {

            case CLOUDLET_SUBMIT_TO_MEC:
                Cloudlet cloudlet = (Cloudlet) ev.getData();
                Log.printLine(CloudSim.clock()+": "+cloudlet.getCloudletId()+"号任务边缘已收到！");
                break;

            default:
                Log.printLine(getName() + ": unknown event type");
                break;
        }
    }

    @Override
    public void startEntity() {
        Log.printLine(super.getName()+" is starting...");
        //schedule(getId(), 0, CREATE_BROKER);
//        for(int i=1;i<500;i++) {
//            schedule(getId(), 100*i, CREATE_BROKER);
//        }
    }

    @Override
    public void shutdownEntity() {}
}
