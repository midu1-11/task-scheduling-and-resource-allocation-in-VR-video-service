# 面向VR视频服务的任务调度与资源分配

## web_vr部分

### 介绍

本部分是一个多用户全景VR视频渲染客户端模拟demo，由主机+树莓派+腾讯云组成，其中主机作为边缘服务器，树莓派作为多用户VR客户端，腾讯云作为云服务器。其中server_cloud文件夹需要保存于云服务器、server_edge文件夹需要保存于边缘服务器、web文件夹需要保存于客户端。

### 运行过程

1. 云服务器

   ```shell
   # 进入云服务器中的项目目录
   cd /home/code/PythonCode/http_server
   # 启动虚拟环境
   workon http_server
   #运行云服务器程序
   python3.9 server_cloud.py
   ```

2. 边缘服务器，运行server_edge.py

3. 客户端主机，用火狐浏览器打开index.html（其他浏览器速度会慢）

4. 选择每个客户端的全景视频、分辨率大小、渲染视角，选择随机均分策略或阈值比例策略或基于强化学习的在线策略

5. 点击提交“所有任务给边缘中心控制器”按钮，在页面下方会显示所有任务的端到端延迟之和，可以用于比较三种策略的效果

6. 当所有客户端都收到任务结果后，可以返回第4步

### BUG

1. 云服务器部分必须要放在真实云服务器上，边缘服务器和客户端部分可以都放在本地网络，或者放在同一个局域网内，不支持云服务器部分也放在局域网或本地网络。
