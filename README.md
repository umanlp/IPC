 <h1 align="center">
<span>Inter-Party Communication (IPC)</span>
</h1>

------------------------
## Repository Description

We present a new method for investigating inter-party communication, based on 
transfer learning, and apply our method to two different scenarios:
1. coalition signal detection
2. negative campaigning, i.e., extracting the target and stance towards this target from party press releases.

This repository contains the code and data needed to reproduce the experiments and results reported in our paper. 

## Negative Campaigning 

### Data 

- **data/negative_campaigning** 
    - This folder contains the following subfolders:
      - **target** 
         - This folder contains the data sets for target prediction, extracted from the AUNTES press releases
      - **stance** 
         - This folder contains the data sets for stance prediction, extracted from the AUNTES press releases


------------------------
#### Citation for AUTNES data

```
@misc{Mueller-etal-2021,
   author = {Müller, Wolfgang C. and Bodlos, Anita and Dolezal, Martin and Eder, Nikolaus and Ennser-Jedenastik, Laurenz and Gahn, Christina and Graf,     Elisabeth and Haselmayer, Martin and Haudum, Teresa and Huber, Lena Maria and Kaltenegger, Matthias and Meyer, Thomas M. and Praprotnik, Katrin and Reidinger, Verena and Winkler, Anna Katharina},
   publisher = {AUSSDA},
   title = {{AUTNES Content Analysis of Party Press Releases: Cumulative File (SUF edition)}},
   UNF = {UNF:6:V9hNiWjjSnOK8j2CFzWZlw==},
   year = {2021},
   notex = {V1},
   doi = {10.11587/25P2WR},
   url = {https://doi.org/10.11587/25P2WR}
}
```
<a href="https://www.autnes.at/autnes-daten/">AUTNES Party Press Releases (original data set)</a>


## TARGET PREDICTION 


### Rule-based dictionary baseline

Change into the dictionary baseline directory and run the python script:

cd IPC/dictionary_baseline

python rulebased_target.py 

This might take a while. The output looks like this:


> Rule-based results for targets:	 fold1 	f1 (macro) 0.48369630013349035 	f1 (micro) 0.6024904214559387
> 
> Rule-based results for targets:	 fold2 	f1 (macro) 0.48178013738918696 	f1 (micro) 0.6338406445837064
> 
> Rule-based results for targets:	 fold3 	f1 (macro) 0.5040863768741828 	f1 (micro) 0.6299151888974557
> 
> Rule-based results for targets:	 fold4 	f1 (macro) 0.43145097724551756 	f1 (micro) 0.5537909836065574
> 
> Rule-based results for targets:	 fold5 	f1 (macro) 0.42835486793855193 	f1 (micro) 0.5619757688723206


### SVM baseline 

Change into the SVM baseline directory and run the python script:

cd IPC/svm_baseline

python svm_baseline_target.py 

This might take a while. The output looks like this:

>[02/Aug/2022 16:36:44] INFO - Start training on: fold1
> 
>[02/Aug/2022 16:38:35] INFO - Train vectorizer and do feature selection: 0.05s
> 
>[02/Aug/2022 16:38:35] INFO - ACC:  0.56s
> 
>[02/Aug/2022 16:38:35] INFO - PREC: 0.55s
> 
>[02/Aug/2022 16:38:35] INFO - REC:  0.50s
> 
>[02/Aug/2022 16:38:35] INFO - F1:   0.47s
> 
>[02/Aug/2022 16:38:35] INFO - Results for LinearSVC	fold1	f1 (macro) 0.47s	f1 (micro)  0.56s
> 
>...


Model predictions are written to:

predictions_SVC-fold1.txt, predictions_SVC-fold2.txt, ..., predictions_SVC-fold5.txt

Results (F1) are written to:

f1_SVC-fold1.txt, f1_SVC-fold2.txt, ..., f1_SVC-fold5.txt



### Transfer learning

**Train and run transfer learning model for target prediction (with early stopping).**

   Attention: model produces many checkpoints => clean up after training to free disk space!


Change into the transfer directory for stance prediction and run the python script:

cd IPC/transfer_target

python transfer_target.py


> Output is written to: outputs.bert-base-german-cased.target.fold[1-5]/
> 
> Best trained model (after early stopping) is in 
>>	- outputs.bert-base-german-cased.target.foldN/best_model
>
> Predictions are written to folder best_model:
>>	- predictions_dev.txt and predictions_test.txt
>
> Results are written to folder best_model:
>>	- results_dev.txt and results_test.txt


After training the model and predicting the labels, you can evaluate the predictions:

python eval_transfer_target_prediction.py



## STANCE PREDICTION  

### Rule-based dictionary baseline

Change into the dictionary baseline directory and run the python script:

cd IPC/dictionary_baseline
 
python rulebased_stance.py

This might take a while. The output looks like this:

> Rule-based results for stance:	 	f1 (macro) 0.26846129713102773 	f1 (micro) 0.40804597701149425
> 
> Rule-based results for stance:	 	f1 (macro) 0.2441511590893233 	f1 (micro) 0.3809310653536258
> 
> Rule-based results for stance:	 	f1 (macro) 0.2455597927104502 	f1 (micro) 0.3831919814957594
> 
> Rule-based results for stance:	 	f1 (macro) 0.2392858045690242 	f1 (micro) 0.3596311475409837
> 
> Rule-based results for stance:	 	f1 (macro) 0.24596685310556862 	f1 (micro) 0.3383038210624418


### SVM baseline 

Change into the SVM baseline directory and run the python script:

cd IPC/svm_baseline
 
python svm_baseline_target.py

This might take a while. The output looks like this:

> [02/Aug/2022 22:13:48] INFO - Time to train vectorizer and do feature selection: 0.08s
> 
> [02/Aug/2022 22:13:49] INFO - ACC:  0.66s
> 
> [02/Aug/2022 22:13:49] INFO - PREC: 0.59s
> 
> [02/Aug/2022 22:13:49] INFO - REC:  0.45s
> 
> [02/Aug/2022 22:13:49] INFO - F1:   0.45s
> 
> [02/Aug/2022 22:13:49] INFO - Results for LinearSVC	fold1	f1 (macro) 0.45s	f1 (micro)  0.66s
> 
> ...


### Transfer learning

Change into the transfer directory for stance prediction and run the python script:

cd IPC/transfer_stance

python transfer_stance.py


After training the model and predicting the labels, you can evaluate the predictions:

Change into the IPC folder and set the seed (an integer between 1 and 5; the specific initialisation that you want to evaluate).

cd ..

python transfer_stance/eval_transfer_target_stance_prediction.py transfer_target/run${seed}_[0-9][0-9]*/outputs.bert-base-german-cased.target.fold$seed/best_model/transfer_predictions_target_test.txt transfer_stance/run${seed}_[0-9][0-9]*/outputs.bert-base-german-cased.stance.fold$seed/best_model/transfer_predictions_stance_test.txt

