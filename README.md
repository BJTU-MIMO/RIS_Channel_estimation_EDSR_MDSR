# RIS_Channel_estimation_EDSR_MDSR

This simulation code package is mainly used to reproduce the results of the following paper [1]:

[1] Y. Jin, J. Zhang, X. Zhang, H. Xiao, B. Ai and D. W. K. Ng, "Channel Estimation for Semi-Passive Reconfigurable Intelligent Surfaces With Enhanced Deep Residual Networks," in IEEE Transactions on Vehicular Technology, vol. 70, no. 10, pp. 11083-11088, Oct. 2021.

If you use this simulation code package in any way, please cite the original paper [1] above. 

Reference: We highly respect reproducible research, so we try to provide the simulation codes for our published papers. 

## Abstract of the paper: 

Reconfigurable intelligent surface (RIS) is envisioned as an essential paradigm for realizing the sixth-generation networks, due to the use of low-cost reflecting elements for establishing programmable and favourable wireless environment. However, accurate channel estimation is a fundamental technical challenge for achieving large performance gains brought by RIS. To address this challenge, we first integrate a RIS with a small number of uniformly distributed active sensing devices, which are equipped with active radio frequency chains for acquiring partial channel state information (CSI). Then, by leveraging the rank-deficient structure of RIS channels, two practical residual neural networks, named single-scale enhanced deep residual (EDSR) and multi-scale enhanced deep residual (MDSR), are proposed to obtain accurate CSI, which can strike a balance between the system complexity and estimation performance. Simulation results reveal the cost-performance trade-off of the two proposed methods and unveil their superior performance compared with existing baseline schemes.

## Content of Code Package

The package generates the simulation results:

- `main.py`: Main function;
- `functional.py`: The used function in the project;
- `model/`: Generate the EDSR or MDSR Network;
- `option.py`: Parameter modification;
- `trainer.py`: Procedures for conducting training EDSR and MDSR;

See each file for further documentation.

## License and Referencing

This code package is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

Enjoy the reproducible research!








