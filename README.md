# IPC
InterPartyCommunication



## TARGET PREDICTION 


### Rule-based dictionary baseline

>cd IPC/dictionary_baseline

>python rulebased_target.py 

>This might take a while. The output looks like this:


Rule-based results for targets:	 fold1 	f1 (macro) 0.48369630013349035 	f1 (micro) 0.6024904214559387

Rule-based results for targets:	 fold2 	f1 (macro) 0.48178013738918696 	f1 (micro) 0.6338406445837064

Rule-based results for targets:	 fold3 	f1 (macro) 0.5040863768741828 	f1 (micro) 0.6299151888974557

Rule-based results for targets:	 fold4 	f1 (macro) 0.43145097724551756 	f1 (micro) 0.5537909836065574

Rule-based results for targets:	 fold5 	f1 (macro) 0.42835486793855193 	f1 (micro) 0.5619757688723206


### SVM baseline 

>cd IPC/svm_baseline
>python svm_baseline_target.py 

This might take a while. The output looks like this:

>[02/Aug/2022 16:36:44] INFO - Start training on: fold1
>[02/Aug/2022 16:38:35] INFO - Train vectorizer and do feature selection: 0.05s
>[02/Aug/2022 16:38:35] INFO - ACC:  0.56s
>[02/Aug/2022 16:38:35] INFO - PREC: 0.55s
>[02/Aug/2022 16:38:35] INFO - REC:  0.50s
>[02/Aug/2022 16:38:35] INFO - F1:   0.47s
>[02/Aug/2022 16:38:35] INFO - Results for LinearSVC	fold1	f1 (macro) 0.47s	f1 (micro)  0.56s
>...


Model predictions are written to:
> predictions_SVC-fold1.txt, predictions_SVC-fold2.txt, ..., predictions_SVC-fold5.txt

Results (F1) are written to:
> f1_SVC-fold1.txt, f1_SVC-fold2.txt, ..., f1_SVC-fold5.txt



### Transfer learning

**Train and run transfer learning model for target prediction (with early stopping).**
   Attention: model produces many checkpoints => clean up after training to free disk space!

cd IPC/transfer
python transfer_target.py

# Output is written to: outputs.bert-base-german-cased.target.fold[1-5]/

# Best trained model (after early stopping) is in 
	- outputs.bert-base-german-cased.target.foldN/best_model

# Predictions are written to folder best_model:
	- predictions_dev.txt and predictions_test.txt

# Results are written to folder best_model:
	- results_dev.txt and results_test.txt



### STANCE PREDICTION ###
#########################

### Rule-based dictionary baseline

cd IPC/dictionary_baseline

python rulebased_stance.py

# This might take a while. The output looks like this:

Rule-based results for stance:	 	f1 (macro) 0.26846129713102773 	f1 (micro) 0.40804597701149425
Rule-based results for stance:	 	f1 (macro) 0.2441511590893233 	f1 (micro) 0.3809310653536258
Rule-based results for stance:	 	f1 (macro) 0.2455597927104502 	f1 (micro) 0.3831919814957594
Rule-based results for stance:	 	f1 (macro) 0.2392858045690242 	f1 (micro) 0.3596311475409837
Rule-based results for stance:	 	f1 (macro) 0.24596685310556862 	f1 (micro) 0.3383038210624418
---

### SVM baseline 

cd IPC/svm_baseline
python svm_baseline_target.py

# This might take a while. The output looks like this:

[02/Aug/2022 22:13:48] INFO - Time to train vectorizer and do feature selection: 0.08s
[02/Aug/2022 22:13:49] INFO - ACC:  0.66s
[02/Aug/2022 22:13:49] INFO - PREC: 0.59s
[02/Aug/2022 22:13:49] INFO - REC:  0.45s
[02/Aug/2022 22:13:49] INFO - F1:   0.45s
[02/Aug/2022 22:13:49] INFO - Results for LinearSVC	fold1	f1 (macro) 0.45s	f1 (micro)  0.66s
...
---

### Transfer learning

cd IPC/transfer
python transfer_target.py


