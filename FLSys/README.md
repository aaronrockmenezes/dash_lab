Federated Learning Based Systems
===============================

I've attempted the following tasks:
Level 1 - Getting Started with Flower
-> I've executed the code as mentioned in the Flower-PyTorch tutorial and added screenshots of the output.

Level 2 - Knowledge Distillation
-> I've implemented simple knowledge distillation using the CIFAR-10 dataset using Cross Entropy loss and distiliing knowledge after each training epoch.

Level 3 - Federated Knowledge Distillation
-> I've implemented the NTDLoss and the FedNTD strategy. 
-> Note that the algorithm mentioned in the paper varies from the author's implementation slightly, wherein the paper talks about KD multiple times in the same epoch, i.e. after each batch, while the author's implementation does it once after each epoch. 
-> I've implemented the algorithm as done by the author, conducting KD once after each epoch.