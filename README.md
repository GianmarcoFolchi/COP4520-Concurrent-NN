# COP4520-Project-1
## How to run: 
Run the following command to install all dependencies `pip install -r requirements.txt`
Execute the XOR.py file for a sample run. 

## Introduction

 Deep learning is a game-changer in many areas, like computer vision, understanding human language, self-driving cars, and more. But, training these complex systems needs a lot of computer power, which can slow things down. Traditional ways of training take too long, especially if you need quick results or want to try different things. Our project plans to speed up training by using parallel computing, which means doing many tasks at the same time across different computer processors. This way, we can make training faster and more efficient, helping make deep learning more widespread and easier to use.

Objectives Our main goal is to create a system for training neural networks in parallel. Here's what we plan to do:

Learn All About Neural Networks and Multithreading: We'll dive deep into how neural networks work and understand parallel computing and multithreading. Knowing these basics is key to using parallel computing for neural network training.

Pick and Set Up a Neural Network Model: We'll use a simple, widely-used model called a feedforward neural network. It will help us test our training methods and how to make them run in parallel.

Concurrency Plan: On each epoch, we will split the training data into N unique groups where N is the number of threads that we want to utilize. Each thread will have a deep copy of the model and will train itself on its unique subset of the data. Once all N models are trained, we aggregate the weights and biases of each model into a single model by taking the average. We will continue to do this for as many epochs as specified on the initial fit call. 

Features and Implementation Details

Simple Neural Network Architecture: Focusing on a feedforward model makes it easier to concentrate on parallelization without the extra complexity of other types of networks. We'll design it to be flexible, testing different setups easily.

Speeding Up Model Development: Faster training lets people experiment and improve AI models quicker, pushing innovation.

Handling Big Data Better: Being able to train with large datasets opens up possibilities for solving more complex problems that were too big to handle before.

Making AI More Accessible: Easier training of advanced neural networks means more people and organizations can use deep learning, even with limited resources.

Conclusion We're aiming to push the limits of deep learning by making an efficient system for training neural networks in parallel. By carefully choosing our models, innovating in how we train, and spreading out the workload, we expect to not only cut down training time but also make deep learning more practical. This project could spark more research and development in AI, showing what's possible with today's technology.

Challenges We Faced

When we started our project, we hit a big roadblock right away: we had to make our own neural network from the very beginning. We wanted to use ready-made tools, but we needed the flexibility to make changes too all aspects of the NN implementation. So, we decided the best move was to create the neural network ourselves, from scratch.

What We Need to Do Next: 
Currently we create the threads and and train each thread seperately, but it is not learning as it should therefore we need to debug.