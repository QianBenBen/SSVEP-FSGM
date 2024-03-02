# SSVEP-FSGM
**My first attemp in BCI adversarial sample.**  
**Here are the command formats for each mode**

## 1、Train Mode
`python main.py --mode train --env_name [your env_name] --model EEGNet --checkpoint best_acc.tar`



## 2、Test Mode
`python main.py --mode test --env_name [your env_name] --model EEGNet --checkpoint best_acc.tar`



## 3、Attack Mode
### 3.1  fgsm attack 
**untargeted attack**

`python main.py --mode attack --env_name [your env_name] --model EEGNet --checkpoint best_acc.tar --epsilon 0.001`

**targeted attack**

`python main.py --mode attack --env_name [your env_name] --model EEGNet --checkpoint best_acc.tar --epsilon 0.001 --target [attack target]`   

### 3.2  i_fgsm attack

**untargeted attack**

`python main.py --mode attack --env_name [your env_name] --model EEGNet --checkpoint best_acc.tar --iteration 5 --epsilon 0.001`

**targeted attack**

`python main.py --mode attack --env_name [your env_name] --model EEGNet --checkpoint best_acc.tar --iteration 5 --epsilon 0.001 --target [attack target]`



## 3.3 My command

`python main.py --mode train --env_name pytorch --model EEGNet --checkpoint best_acc.tar`



`python main.py --mode test --env_name pytorch --model EEGNet --checkpoint best_acc.tar`



`python main.py --mode attack --env_name pytorch --model EEGNet --checkpoint best_acc.tar --epsilon 0.001`



`python main.py --mode attack --env_name pytorch --model EEGNet --checkpoint best_acc.tar --epsilon 0.001 --target 5`   



`python main.py --mode attack --env_name pytorch --model EEGNet --checkpoint best_acc.tar --iteration 5 --epsilon 0.001`



`python main.py --mode attack --env_name pytorch --model EEGNet --checkpoint best_acc.tar --iteration 5 --epsilon 0.001 --target 5`