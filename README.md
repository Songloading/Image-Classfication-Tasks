# Image-Classfication-Tasks

The goal for this repo is to keep a record of COMP_SCI 499 Project for each week and save useful materials online.


### **Week 1**
- **Learning Objectives:** Basic image classification tasks with simple models using **Pytorch** and **Jax**
- **Learning Outcomes:** Finish using Pytorch and Jax to build a Lenet-5 model to do image classification tasks.

- **Findings & Conclusion:**

| Framework/Diff     | Model Bulding     | Model Training     |
| ---------- | :-----------:  | :-----------: |
| Pytorch    | - Model is a class <br> - Define each layer as variables  <br> - Foward Function manually goes through each layer   | - Has dataloader that can be enumerate <br> - Usually use built-in loss function <br>  - Usually we call **optimizer.step()** to update   |
| Jax    | - Model is treated as a function <br> - **stax** for example returns model parameters and foward function     | - Has to define data_stream method <br>   - Has to self-define loss function <br> - Update each batch state and pass to the next   |
