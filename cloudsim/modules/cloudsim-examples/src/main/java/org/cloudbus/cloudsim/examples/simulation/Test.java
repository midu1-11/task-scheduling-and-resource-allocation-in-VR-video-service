package org.cloudbus.cloudsim.examples.simulation;

import org.cloudbus.cloudsim.core.CloudSim;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

/**
 * @author howard
 * @version 1.0
 */
public class Test {
    public static void main(String[] args) throws Exception {
        int n = 25;
        boolean isSaveDelay = false;
        boolean isShare = true;
        int num_user = 1;   // number of grid users
        int cloudletIdShift = 0;
        Calendar calendar = Calendar.getInstance();
        boolean trace_flag = false;  // mean trace events

        // Initialize the CloudSim library
        CloudSim.init(num_user, calendar, trace_flag);

        EdgeServer edgeServer = EdgeServer.createEdgeServer("edgeServer",isShare);
        CloudServer cloudServer = CloudServer.createCloudServer("cloudserver");

        Controller controller = new Controller("controller", edgeServer, cloudServer, isSaveDelay);

        List<Client> ClientList = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            ClientList.add(new Client("client" + i, controller));
        }

        CloudSim.startSimulation();
    }
}
