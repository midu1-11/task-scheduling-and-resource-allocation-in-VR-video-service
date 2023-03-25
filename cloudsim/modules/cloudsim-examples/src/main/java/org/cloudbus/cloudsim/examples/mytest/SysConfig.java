package org.cloudbus.cloudsim.examples.mytest;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Properties;

/**
 * @author howard
 * @version 1.0
 */
public class SysConfig {

    private static Properties properties;

    static {
        properties = new Properties();
        try {
            properties.load(new InputStreamReader(new FileInputStream("F:\\Bupt\\cloudsim\\modules" +
                    "\\cloudsim-examples\\src\\main\\java\\org\\cloudbus\\cloudsim\\examples\\mytest" +
                    "\\configuration.properties")));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static String getInfo(String key) {
        return properties.getProperty(key);
    }
}
