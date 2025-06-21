
## Training plan

- Environment: PC, Google Colab, Kaggle
- Load the data
- Standardize all input to a fixed size (eg.: 100x100)
- Preprocesare: normalization, reshaping, tensoring, convert categorical to numerical values, fast fourier transformation
- Augmentation: Apply transformations such as horizontal/vertical flips, rotations, zooming, and brightness adjustments to increase data diversity
- Train-test split logic: cross validation cu k-fold, train - test - validate split & check for balance in train / test / validate datasets
- Load the data in batches: DataLoaders & batch size & shuffling
- Definirea structurii retelei
  - Numarul initial de canale
  - Numarul de straturi convolutionale: the more layers, the stronger regularization
  - Pentru fiecare strat convolutional: kernel size, channels out, padding, stride
  - Pentru fiecare strat de pooling: kernel size
  - BatchNorm2D
  - Definirea functiilor de activare pentru fiecare strat convolutional
  - Weights initialisation: Xavier, all weights with 1
  - Global Average Pooling to reduce parameters and overfitting
- Optimization algorithm: adam, sgd
- Loss function: categorical cross entropy
- Stabilirea numarului de epoci
- Gradient checking
- Regularization: weight decay with L2 and small coefficient (1e-4), early stopping, dropout
- Stabilire learning rate: exponential decay, step decay, 1/t decay

- **Fine tunning**: experiment with different parameters and save metrics for final report & ask for help

- Evaluation Metrics
  - Report confusion matrices for the provided validation set with various models
  - Train loss & Validation loss
  - Learning rate variability
  - Other hyperparameters
