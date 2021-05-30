# Image-Classfication-Tasks

The goal for this repo is to keep a record of COMP_SCI 499 Project for each week and save useful materials online.


### **Week 1**
- **Learning Objectives:** Basic image classification tasks with simple models using **Pytorch** and **Jax**
- **Learning Outcomes:** Finish using Pytorch and Jax to build a Lenet-5 model to do image classification tasks.

- **Findings & Conclusion:**

| Framework/Diff     | Model Bulding     | Model Training     | Others |
| ---------- | :-----------:  | :-----------: | :-----------: |
| Pytorch    | - Model is a class <br> - Define each layer as variables  <br> - Foward Function manually goes through each layer   | - Has dataloader that can be enumerate <br> - Usually use built-in loss function <br>  - Usually we call **optimizer.step()** to update   | - Provide plenty of datasets |
| Jax    | - Model is treated as a function <br> - **stax** for example returns model parameters and foward function     | - Has to define data_stream method <br>   - Has to self-define loss function <br> - Update each batch state and pass to the next   | - Has to manually define dataset |


### **Week 2**
- **Learning Objectives:** Getting started on Julia and corresponding libraries.
- **Material**: 
- - https://syl1.gitbook.io/julia-language-a-concise-tutorial/language-core/getting-started
- - https://github.com/denizyuret/Knet.jl/tree/master/tutorial
- **Learning Outcomes:** Finish using Julia and Knet building Lenet5 and classify MNIST data.
- **Findings & Conclusion:**
- - Julia is very similar to Python
- - Package Reference is not very clear
- - It can be very easy and clear to build each layer of NN by using the Julia Constructor

### **Week 3**
- **Learning Objectives:** Try to load self-defined data to each of the three previous learned pipeline.
- **Learning Outcomes:** Finish using Julia, Pytorch, and Knet to load custome dataset.

### **Week 4-10: Classify the X-Ray dataset using different models w/ high accuracies**
- **Outline:** The
- **Reference & Resources:**
  <br /> https://www.kaggle.com/nih-chest-xrays/data 
  https://jrzech.medium.com/reproducing-chexnet-with-pytorch-695ff9c3bf66
