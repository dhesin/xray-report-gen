# xray-report-gen

1. xray_enc_dec_train.py used the train the model. Outputs are written to xray_model_lr1-3.pth by default.

2. xray_report_gen.py reads the xray_model_lr1-3.pth and generates reports starting with the <start> token and picking the word with the highest probability. It creates reports_preds_generated.csv and preds_only_generated.csv files. reports_preds_generated.csv includes ground truth report and generated report. preds_only_generated.csv includes only generated report for CheXpert Labeler use. 

3. compare_labels.py first runs CheXpert Labeler and compared the ground truth labels with the labels from generated reports. 

4. dataset.py generates ground truth labels, scaled/normalized images and labels during training/testing from given dataframes.

5. gen_vocab_datasets.py creates vocabulary/tokens and splits UI X ray into train and test set.

6. trainer.py handles the training/testing loop with dataloaders, learning rate scheduling, optimizer and back propagation. 

7. utils.py has seed setting and sampler code for text generation

8. mymodel.py defines the architecture

8. tokenizer.py is used to experiment with WordPieceTokenizer from BERT but was not promising. 

We have used some starter code from https://github.com/karpathy/minGPT mostly trainer.py, utils.py. But most of our code including the model architecture iswritten by our team.
