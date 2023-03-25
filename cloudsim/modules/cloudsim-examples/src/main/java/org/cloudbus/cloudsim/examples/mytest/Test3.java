package org.cloudbus.cloudsim.examples.mytest;

import javafx.scene.chart.XYChart;

import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;

import java.io.*;
import java.util.ArrayList;
import java.util.List;


/**
 * @author howard
 * @version 1.0
 */
public class Test3 {
    public static void main(String[] args) {
        Utils.printResultChart(MyStatement.ALGORITHM_RESULT);
        Utils.printResultChart(MyStatement.BASELINE_RESULT);
    }
}
