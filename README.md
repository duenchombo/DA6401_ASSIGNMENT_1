# Assignment 1 : Training a feedforward neural network

**NOTE :** The program is written in a modular manner, with each logically separate unit of code written as functions.  


- The solution to each question is made in the form of a couple of function calls (which are clearly mentioned with question numbers in comments) and commented out in the program so that the user can choose which parts to run and evaluate.
- In order to run the solution for a particular question, uncomment that part and run the notebook.
- By default, the solution for question 7, to train the neural network for the configuration giving best accuracy and plotting it's confusion matrix is uncommented.
- In order to train a new neural network on custom configurations and dataset, call :
  ```python
  train_wrapper(trainX, trainY, optimizer, batch_size, learning_rate, max_epochs, no_hidden_layers, 
                size_hidden_layer, weight_initialisation, activation, loss, weight_decay = 0,
                validX = None, validY = None, testX = None, testY = None, regularisation = 'L2')
  ```
  where the arguments are :
  ```
  trainX -- (matrix) Input training data matrix with data as column vectors
  trainY -- (matrix) True training output data matrix with data as column vectors
  optimizer -- (string) Optimisation function. Takes values only in ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']
  batch_size -- (int) Batch size for training
  learning_rate -- (float) Hyperparameter eta for gradient terms in training
  max_epochs -- (int) Maximum number of epochs to train the neural network
  no_hidden_layers -- (int) Number of hidden layers in the neural network
  size_hidden_layer -- (int) Number of neurons in each hidden layer
  weight_initialisation -- (string) Weight initialisation method. Takes values only in ['random', 'xavier']
  activation -- (string) Activation function for each neuron. Takes values only in ['relu', 'sigmoid', 'tanh]
  loss -- (string) Loss function used. Takes values only in ['cross-entropy', 'squared-error']
  weight_decay -- (float) Hyperparameter lambda for regularisation term in training
  validX -- (matrix) Input validation data matrix with data as column vectors
  validY -- (matrix) True validation output data matrix with data as column vectors
  testX -- (matrix) Input testing data matrix with data as column vectors
  testY -- (matrix) True testing output data matrix with data as column vectors
  regularisation -- (string) Type of regularisation used. Takes values only in ['L2', 'L1']
  ```
  and the function returns :
  ```
  train_stats -- List of tuple (training accuracy, average training loss) of entire dataset after every epoch of training
  valid_stats -- List of tuple (validation accuracy, average validation loss) of entire dataset after every epoch of training
  test_stat -- Tuple (test accuracy, average test loss) of entire dataset for the trained model
  ```
  All the hyparameters, loss and accuracy are also logged in **WANDB**.
- In order to run a new sweep in **WANDB**, run :
  ```python
  sweep_id = wandb.sweep(project="assignment_1", entity="duenchombo1-indian-institute-of-technology-madras")
  wandb.agent(sweep_id, lambda : sweep_wrapper(trainX, trainY, validX, validY, testX, testY, loss))
  ```
  where the arguments to `sweep_wrapper` take the same meaning as in  `train_wrapper` above. The sweep with all metrics and configurations is logged in **WANDB**.
  The `sweep_config` argument is a dictionary containing the method of search, goal to optimize, and parameters to sweep. An example from the code itself is :
  ```python
  sweep_config = {
      'method': 'bayes',                   # Possible search : grid, random, bayes
      'metric': {
        'name': 'validation_accuracy',
        'goal': 'maximize'   
      },
      'parameters': {
          'no_epochs': {
              'values': [5, 10]
          },
          'no_hidden_layers': {
              'values': [3, 4, 5]
          },
          'size_hidden_layer': {
              'values': [32, 64, 128]
          },
          'weight_decay' :{
              'values': [0, 0.0005, 0.005]
          },
          'learning_rate': {
              'values': [1e-1, 1e-2, 1e-3]
          },
          'optimizer': {
              'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam' ]
          },
          'batch_size': {
              'values': [16, 32, 64]
          },
          'weight_initialisation': {
              'values': ['random', 'xavier']
          },
          'activation_fn': {
              'values': ['relu', 'tanh', 'sigmoid']
          }
      }
  }
  ```
  The above dictionary can be changed according to needs and used to obtain a different sweep.
- **NOTE :** Always run the cells in order, otherwise incorrect results may be obtained
