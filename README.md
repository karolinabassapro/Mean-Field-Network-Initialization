# TDL-Initialization-Project

TODO:
- Implement function for plotting against other models
- Hyperparameter tuning for learning rate and (maybe) num conv channels in rectangular region
- Compare models at different depth scales
- Decide on graphs to be produced to run on GPUs
- Refactor CNN?

WARNING: there are two parameters "name" and "q" in Init_specifications, the former of which must match the value passed into MeanField get_h0 slot 1 when called and using the Gaussian initialization. name will also control which activation function is considered. This is an ad hoc solution which should be remedied soon
