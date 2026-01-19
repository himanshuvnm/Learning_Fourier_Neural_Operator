# Learning Fourier Neural Operator

For datasets use: https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-

This repo reproduces and analyzes a Fourier Neural Operator (FNO) baseline on the 1D viscous Burgers' equation at Reynolds number Re=10.  

## What I implemented
- Data preprocessing + train/val/test splits
- FNO model + training loop
- Evaluation on held-out trajectories
- Resolution study with consistent hyperparameters

<img width="1678" height="1302" alt="64resolution" src="https://github.com/user-attachments/assets/0233e7bb-7a72-4ef2-bf06-8da1a0c2787e" />

<img width="1678" height="1302" alt="256resolution" src="https://github.com/user-attachments/assets/b34dda6e-1611-4420-86fa-27b447b8cfc4" />

<img width="1678" height="1302" alt="1024resolution" src="https://github.com/user-attachments/assets/8211264a-a058-40fa-bdeb-51aa0a4a1579" />

<img width="3034" height="1113" alt="FNO_Berger_Learning" src="https://github.com/user-attachments/assets/e9ded1cd-8b0e-4512-ad22-26d08df4e17c" />
