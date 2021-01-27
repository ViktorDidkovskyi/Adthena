import mlflow.sklearn
import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# using scikit learn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from utils.tfidf_vectorizer import TfidfLocVectorizer
from utils.clean_text_transformer import CleanTextTransformer
from utils.dimension_reduction import TruncatedLocSVD
from utils.preprocessor import Preprocessor
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier
import lightgbm as lgb

import yaml

RANDOM_STATE = 42


class ClassifierModel:
    """The model which helps in identifying the search term classes"""

    def __init__(self,
                 input_data=None,
                 tfidf_max_feat=500,
                 mlflow_local=False,
                 classifier_type: str = 'lgbm',
                 use_data_under_balancer: bool = False,
                 vectorizer_type: str = 'tfidf',
                 optimum_n_components: int = None,
                 pipeline=None,
                 mlflow_local_url='mlruns_new',
                 model_save_loc='Model_Save'):

        """
        Constructor for classifier
        :param input_data:
        :param tfidf_max_feat:
        :param mlflow_local:
        :param classifier_type:
        :param use_data_under_balancer:
        :param vectorizer_type:
        :param optimum_n_components:
        :param pipeline:
        :param mlflow_local_url:
        :param model_save_loc:
        """

        self.mlflow_local_url = mlflow_local_url
        self.model_save_loc = model_save_loc

        self.text_cleaner = CleanTextTransformer()
        self.classifier = None
        self.classifier_type = classifier_type
        self.vectorizer = None
        self.vectorizer_type = vectorizer_type
        self.pipeline = pipeline
        self.use_data_under_balancer = use_data_under_balancer
        self.data_under_balancer = None
        self.dimension_reduction = None
        self.optimum_n_components = optimum_n_components

        self.max_feat = tfidf_max_feat

        self.x_train = self.y_train = self.x_test = self.y_test = None
        self.input_data = input_data
        self.is_local = mlflow_local

    def __training_setup(self, input_data):
        """ Method to initialize all the sub models/objects used as part of the classifier model"""
        logger.info("Setting up model for classifier")
        # Get Data if provided

        self.preprocessor = Preprocessor(input_data)
        self.x_train, self.x_test, self.y_train, self.y_test = self.preprocessor.get_data()

        logger.info("Setting up Vectorizer")
        # Vectorizer
        if self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfLocVectorizer(max_feat=self.max_feat, maxdf=0.8,
                                                 mindf=15, n_gram_range=(1, 3))

        elif self.vectorizer_type == 'spacy':
            import spacy
            from utils.spacy_vectorizer import SpacyVectorTransformer
            nlp = spacy.load("en_core_web_md")
            self.vectorizer = SpacyVectorTransformer(nlp=nlp)
        else:
            raise ValueError("incorrect vectorizer_type, please use tfidf or spacy")
        # Balance the data
        if self.use_data_under_balancer:
            logger.info("Setting up Naive Balance the data")

            self.data_under_balancer = RandomUnderSampler(sampling_strategy=
                                                          {l: min(70, number - 1) for l, number in
                                                           self.y_test.value_counts().items()})

        logger.info("Run dimension reduction algorithm")
        self.dimension_reduction = TruncatedLocSVD(self.optimum_n_components, total_variance=0.8)

        logger.info("Setting up Classifier")
        # Classifier
        if self.classifier_type == 'xgb':
            self.classifier = XGBClassifier(colsample_bytree=0.7, learning_rate=0.05, max_depth=5,
                                            min_child_weight=11, n_estimators=1000, n_jobs=4,
                                            objective='binary:multiclass', random_state=RANDOM_STATE, subsample=0.8)
        elif self.classifier_type == 'lgbm':
            params = {'num_leaves': 5,
                      'objective': 'multiclass',
                      'num_class': len(np.unique(self.y_train)),
                      'learning_rate': 0.01,
                      'max_depth': 5,
                      'random_state': RANDOM_STATE
                      }
            self.classifier = lgb.LGBMClassifier(**params)

        else:
            self.classifier = LogisticRegression(multi_class="multinomial",
                                                 class_weight='balanced',
                                                 solver='newton-cg',
                                                 max_iter=100)

        # MLFlow Config
        logger.info("Setting up MLFlow Config")
        mlflow.set_experiment('classifier-model')

    def train(self, train_x=None, train_y=None, grid_search=True, run_version=None):
        """ Method to train the model
        :param train_x: independent data to train model with
        :param train_y: dependent data to train model with
        :param grid_search: perform grid_search
        :param run_version: Load previous run_version for training if needed
        :return: None
        """
        self.__training_setup(self.input_data)

        logger.info("Training for search term classifier model")
        if not train_x:
            train_x = self.x_train
            train_y = self.y_train

        # Search for previous runs and get run_id if present
        logger.info("Searching for previous runs for given model type")
        df_runs = mlflow.search_runs(filter_string="tags.Model = '{0}'".format('XGB'))
        df_runs = df_runs.loc[~df_runs['tags.Version'].isna(), :] if 'tags.Version' in df_runs else pd.DataFrame()
        if not run_version:
            run_id = None
            load_prev = False
        else:
            try:
                run_id = df_runs.loc[df_runs['tags.Version'] == run_version, 'run_id'].iloc[0]
                load_prev = True
            except Exception as e:
                raise ValueError('run_id with version {0} not found'.format(run_version))
        run_version = len(df_runs) + 1

        # Start the MLFlow Run and train the model
        logger.info("Starting MLFlow run to train model")
        with mlflow.start_run(run_id=run_id):
            # Build pipeline. Load previous pipeline if needed
            if load_prev:
                artifact_uri = mlflow.get_artifact_uri(self.model_save_loc)
                try:
                    load_pipeline = mlflow.sklearn.load_model(artifact_uri)
                    self.pipeline = load_pipeline
                except Exception as e:
                    raise ValueError("Existing model not found / couldn't be loaded.\n" + str(e))
            else:
                if self.use_data_under_balancer:
                    self.pipeline = Pipeline([('clean_text', self.text_cleaner),
                                              (self.vectorizer_type, self.vectorizer),
                                              ('balancer', self.data_under_balancer),
                                              ('dimension_reduction', self.dimension_reduction),
                                              (self.classifier_type, self.classifier)])
                else:
                    self.pipeline = Pipeline([('clean_text', self.text_cleaner),
                                              (self.vectorizer_type, self.vectorizer),
                                              ('dimension_reduction', self.dimension_reduction),
                                              (self.classifier_type, self.classifier)])
                # Todo: Grid Search for LGBM
                if grid_search:
                    xgb_parameters = {
                        'clf__njobs': [4],
                        'clf__objective': ['multiclass'],
                        'clf__learning_rate': [0.05],
                        'clf__max_depth': [6, 12, 18],
                        'clf__min_child_weight': [11, 13, 15],
                        'clf__subsample': [0.7, 0.8],
                        'clf__colsample_bytree': [0.6, 0.7],
                        'clf__n_estimators': [5, 50, 100, 1000],
                        'clf__missing': [-999],
                        'clf__random_state': [RANDOM_STATE]
                    }
                    if self.use_data_under_balancer:
                        xgb_pipeline = Pipeline([('clean_text', self.text_cleaner),
                                                 (self.vectorizer_type, self.vectorizer),
                                                 ('balancer', self.data_under_balancer),
                                                 ('dimension_reduction', self.dimension_reduction),
                                                 ('clf', XGBClassifier())])
                    else:
                        self.pipeline = Pipeline([('clean_text', self.text_cleaner),
                                                  (self.vectorizer_type, self.vectorizer),
                                                  ('dimension_reduction', self.dimension_reduction),
                                                  (self.classifier_type, self.classifier)])

                    self.pipeline = GridSearchCV(xgb_pipeline, xgb_parameters, n_jobs=1, verbose=2, refit=True,
                                                 cv=StratifiedKFold(n_splits=3, shuffle=True))

            # Train the model
            self.pipeline.fit(train_x, train_y)
            logger.info("train is done")
            train_pred = self.pipeline.predict(train_x)

            # read the dict with correct labels
            with open('data/relabeling_dict.yml', 'r') as f:
                relabeling_dict = yaml.load(f)
            labeling_dict = dict(map(reversed, relabeling_dict.items()))

            # classification report on train set
            df = pd.DataFrame(classification_report(train_y, train_pred, output_dict=True)).transpose()
            logger.info("test is done")
            # Save tags and model metrics
            logger.info("Training Complete. Logging results into MLFlow")

            mlflow.log_metric("insam_macro_f1", np.round(df.loc["macro avg", "f1-score"], 5))
            mlflow.log_metric("insam_weighted_f1", np.round(df.loc["weighted avg", "f1-score"], 5))
            df = df.reset_index()
            df.columns = ['category', 'precision', 'recall', 'f1-score', 'support']
            df.loc[:, 'category'] = df['category'].apply(lambda x: labeling_dict[eval(x)] if x.isdigit() else x)
            df.to_csv("insam_full_report.csv")
            mlflow.log_artifact("insam_full_report.csv")
            os.remove("insam_full_report.csv")
            # Log params
            if self.classifier_type in ('lgbm', 'xgb'):
                if grid_search:
                    mlflow.log_param("Best Params", self.pipeline.best_params_)
                    mlflow.log_param("Best Score", self.pipeline.best_score_)
                else:
                    params = self.classifier.get_xgb_params() if self.classifier_type == 'xgb' \
                        else self.classifier.get_params()
                    for key in params:
                        mlflow.log_param(key, params[key])
            else:
                mlflow.log_param('class_weight', 'balanced')
                mlflow.log_param('solver', 'newton-cg')
                mlflow.log_param('max_iter', 100)

            if len(self.x_test):
                test_pred = self.pipeline.predict(self.x_test)
                # classification report on test set
                test_df = pd.DataFrame(classification_report(self.y_test, test_pred, output_dict=True)).transpose()

                mlflow.log_metric("macro_f1", np.round(test_df.loc["macro avg", "f1-score"], 5))
                mlflow.log_metric("weighted_f1", np.round(test_df.loc["weighted avg", "f1-score"], 5))
                test_df = test_df.reset_index()
                test_df.columns = ['category', 'precision', 'recall', 'f1-score', 'support']
                test_df.loc[:, 'category'] = test_df['category'].apply(lambda x: labeling_dict[eval(x)] if x.isdigit() else x)
                test_df.to_csv("full_report.csv")
                mlflow.log_artifact("full_report.csv")
                os.remove("full_report.csv")

            mlflow.sklearn.log_model(self.pipeline, self.model_save_loc, serialization_format='pickle')
            mlflow.set_tag("Model", self.classifier_type)
            mlflow.set_tag("Version", run_version)
            logger.info("Model Trained and saved into MLFlow artifact location")

    def predict(self, data_x=None, proba=False):
        """ Method to use the model to predict
        :param data_x: input
        :param proba: result is probability
        :return:
        """
        logger.info("Predicting using classifier model")
        data_x = data_x.loc[:, 'search_term']

        if not proba:
            test_pred = self.pipeline.predict(data_x)

        else:
            dis_index = list(self.pipeline.classes_).index('category')
            test_pred = [x[dis_index] for x in self.pipeline.predict_proba(data_x)]
        return test_pred
