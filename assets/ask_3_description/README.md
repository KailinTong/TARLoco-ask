The Lidar mid360 frame relative to the body frame satisfies the following homogeneous transformation matrix
```
B^T_L = [0  0 -1  0.4104
         1  0  0  0
         0 -1  0  0.017
         0  0  0  1     ] 
```
If you want to transfer a point in the Lidar frame to the base frame, please use the following equation
```
p_B = B^T_L * p_L
```
