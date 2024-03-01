# SSVEP-FSGM
My first attemp in BCI adversarial sample.
## 1、Train Mode
if you want to train a mode, you can use the command  
python main.py --mode train --env_name [your env_name] --model EEGNet
## 2、Test Mode
if you want to test a mode, you can use the command  
python main.py --mode test --env_name [your env_name] --model EEGNet --checkpoint best_acc.tar
## 3、Attack Mode
if you want to attack a mode, you can use the command 
python main.py --mode attack --env_name [your env_name] --model EEGNet --checkpoint best_acc.tar