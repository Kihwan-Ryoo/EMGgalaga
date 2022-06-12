This is the project to control the famous 2D game 'galaga' by using EMG of forearms

On the paper I referenced, 8 channels at each forearm are used. But only 2 channels at each forearm is allowed in this project.  
So, I made ModelSelection.py to select 2 determinants among 8 channels by analyzing the dataset.  
ModelSelection.py gave me box plots telling that channel2 is most important on Wrist Flexion, and channel5 is most important on Wrist Extension.

MSP430firmware.c will be written to MSP430 board. It will detect EMG signal with 12 bits ADC, and send it to PC.  
EMGgalaga.py will read the EMG data, classify it, control player on 'galaga'. It shows, with GUI, the EMG data and connected 'galaga' game. You can check it with the demonstration video (mp4).

Reference:
Lobov, S., et al. Latent Factors Limiting the Performance of sEMG-Interfaces, Sensors, 2018.

Dataset:
Supported by the Ministry of Education and Science of the Russian Federation in the framework of megagrant allocation in accordance with the decree of the government of the Russian Federation â„–220, project â„– 14.Y26.31.0022
