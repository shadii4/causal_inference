# causal_inference

To run the experiment the MIMIC III datasets should be downloaded. If so, define the parameters in config.yaml file to run the experiment with the specified treatment and outcome.
after installing all the required libraries, run create_features.py to preprocess the data then run_exp_bert.py to train the model and estimating the causal effect. 
The estimations and results will be stored in features and results folders. 
