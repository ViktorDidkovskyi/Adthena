# Classifier search terms

This project contained a pipeline to train the model which will classify the search terms.
## Details

The input is URLs for train.csv and test.txt. Train.csv contains search terms and target category 
that was already encoded.
Overall, 606,823 examples and 1419 different classes.
All data locates in the "data" folder. Also, will automatically generate if the input is a new URL.
The prediction on test for English:
``
/Users/viktor/Jobs/Adthena/data/candidateTestSet_en.csv
``
The final model, I put in folder "model" from "mlruns/1/../artifacts/Model_Save"

## Challenging

- Multinomial classification problem with a huge amount of classes - 1419.
- Significant imbalance in classes ( for class 587 - just 4 examples).
- The target category has already been encoded, 
  so impossible to group the target category into clusters.
- Search terms have different languages - 32.
- Not enough capacity to train a huge model and fast-iterate the experiments. 

## Result and pipeline

To overcome all challenges were implement a naive pipeline:


1) Group category which appears less than 50 in one class "-1"
2) Train the model per languages separate( was done only for 'en',
  not enough capacity to train one model; for other languages the pipeline the same).
  Each language has its own specific.
3)  Clean up the input keywords and split them to train/test.
4) Use a simple TF-IDF vectorizer to fast convert data into numbers 
  ( in general independent from languages)  
5) Use undersampling to limit the maximum amount of data for each class - 70, 
  to reduce the amount of data and balance the classes.
6) Use the SVD algorithm to reduce the dimensionality to make the training easier
7) Use the LGBM algorithm to increase the quality. 
  Also, in the pipeline possible to use XGBoost and logistic regression.

The target metrics would be f1-weighted and f1-macro.
Result for en-models I got ~0.2. 
details:
```
mlruns/1/6ceec2f3c2694f9cbbc5f14b62149a16/artifacts/full_report.csv
mlruns/1/6ceec2f3c2694f9cbbc5f14b62149a16/artifacts/insam_full_report.csv
```
The number is so low because the target classes very different, 
so it would be hard to have just one model for all classes in my machine. 
When I have dummy trained on 5 random classes the quality was ~ 0.78.

## Next steps and ways to improve

Most of the experiments I have not done due to a leak of time and not enough capacity.
For each run, I spent the night.


1) It would be great to reduce the number of classes. 
   So, it would be great to group the data into a cluster. 
   If we have real classes we will able to create a cluster on it. 
   As the result, we will have a sequence of classifiers (first predict cluster, then class).
    
2) Use another vectorizer. In pipeline exist spacy, 
   but also it is possible to Fasttext and maybe more sophisticated - BERT, ELMO.
   
3) Add more features and improve preprocessing part.

4) Use other dimension reduction algorithms such as PCA, UMAP, T-SNE

5) Implement Bayesian hyper optimization for LGBM ( I have added just GRID Search for XGB).

6) Group the classes using classifier probability and errors, cross modeling testing.

7) Refactoring, and put the model parameters in separate file and setup.py, etc.

8) Implement another logic for balance the classes. Experiment, with a different strategy.

9) Use deep learning models.

10) Try to use https://github.com/Adapter-Hub/adapter-transformers to train one model for all languages.

11) Translate all languages into English.

12) Group similar languages in one training set.


## How to use
To run the training pipeline, use:
```
python main.py --type_of_run 'train' --select_lang 'en' 
```
Or use model "run_main.py". The default value for urls:
```
--train_url https://s3-eu-west-1.amazonaws.com/adthena-ds-test/trainSet.csv
--test_url https://s3-eu-west-1.amazonaws.com/adthena-ds-test/candidateTestSet.txt
```
To run the prediction pipeline, use:
```
python main.py --type_of_run 'predict' --model_to_load 'classifier_search_terms' --select_lang 'en' 
```

