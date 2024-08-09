# FELEMN: Toward Efficient Feature-level Machine Unlearning for Exact Privacy Protection

### Setup

* We tested the code with `python3.8.7`
* Install PyTorch and related dependencies.



### Code

*FELEMN
*Input the model architectures, the number of sub-models, the dataset, Epoch, Opt, and the batch size, and then call the FELEMN as shown in main_FELEMN.
* The methods GreedyMax and GreedyMin directly call GreedyMax.py and GreedyMin.py.

* S-FELEMN
* Input the model architectures, the number of sub-models, the dataset, Epoch, Opt, and the batch size.
* Create the unlearning and testing requests.
* Call the S-FELEMN(hardvoting) and the baselines OBO(hardvoting), Eraser as shown in main_hardvoting.py.
  
* S-FELEMN_delta
* Input the model architectures, the number of sub-models, the dataset, Epoch, Opt, and the batch size.
*  Create the unlearning and testing requests.
* Call the S-FELEMN_delta(softvoting) and the baselines OBO(softvoting), S-FELEMN(softvoting) as shown in main_softvoting.py.





