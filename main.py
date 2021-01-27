from langdetect import detect
from langdetect import lang_detect_exception

from classifier import logger

import requests
import pandas as pd

import _pickle as cPickle

from os.path import isfile, join
import argparse

from classifier import ClassifierModel
import yaml


def download(url: str):
    """
    Download date from url
    :param url: url for data file
    :return:
    """
    get_response = requests.get(url, stream=True)
    file_name = url.split("/")[-1]
    with open(f"data/{file_name}", 'wb') as f:
        for chunk in get_response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    logger.info(f"the data was saved to file data/{file_name}")



def detect_lang(text: str) -> str:
    """
    Identify language to input text
    :param text:
    :return:
    """
    try:
        lang = detect(text)
    except lang_detect_exception.LangDetectException:
        lang = 'num'
    return lang


def make_dataframe_and_add_languages(input_file: str) -> pd.DataFrame:
    """
    Add headers to the data and language column and save them to the data folder
    :param input_file: input file name
    :return:
    """

    input_df = pd.read_csv(f"data/{input_file}", names=['search_term', 'category'])

    input_df.loc[:, 'language'] = input_df.search_term.apply(detect_lang)

    print(f"all language that has found: {input_df.language.value_counts()}")
    # store with language
    input_df.to_csv(f'data/{input_file.split(".")[0]}_w_lang.csv', index=False)

    return input_df


def naive_group_category(train_df: pd.DataFrame, select_lang: str = None) -> pd.DataFrame:
    """
    Group categories that have less representation than 50 and filter data frame if select_lang.
    Also, relabeling category from 0.
    :param train_df: input train data frame
    :param select_lang: select language for training, default is None
    :return:
    """
    # group category with low representation
    train_df.loc[:, 'default_category'] = train_df.loc[:, 'category']
    category_distribution = train_df.category.value_counts()
    small_category = list(category_distribution[category_distribution < 50].index)
    train_df.loc[:, 'category'] = train_df.category.apply(lambda x: x if x not in small_category else -1)

    if select_lang:
        train_df = train_df[train_df['language'] == select_lang]
    # encoding the classes
    relabeling_dict = {l: i for i, l in enumerate(sorted(train_df.category.drop_duplicates()))}
    train_df.loc[:, 'category'] = train_df.category.apply(lambda x: relabeling_dict[x])

    # Todo: Should be move in another place
    with open('data/relabeling_dict.yml', 'w') as outfile:
        yaml.dump(relabeling_dict, outfile, default_flow_style=False)

    return train_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Classified search terms')

    parser.add_argument('--train_url',
                        help='Url with training data',
                        default='https://s3-eu-west-1.amazonaws.com/adthena-ds-test/trainSet.csv')

    parser.add_argument('--test_url',
                        help='Url with test data',
                        default='https://s3-eu-west-1.amazonaws.com/adthena-ds-test/candidateTestSet.txt')

    parser.add_argument('--type_of_run',
                        help='Type of running mode',
                        default='train')

    parser.add_argument('--model_to_load',
                        help='Model to load if exist',
                        default=None)

    parser.add_argument('--select_lang',
                        help='Selected one language to train/test',
                        default=None)
    args = parser.parse_args()

    train_url = args.train_url
    test_url = args.test_url

    train_file = train_url.split('/')[-1]
    final_test_file = test_url.split('/')[-1]
    # if no data, so download
    if not isfile(f"data/{train_file}"):
        download(args.train_url)
    if not isfile(f"data/{final_test_file}"):
        download(args.test_url)

    if args.model_to_load:
        pipeline = cPickle.load(open(f"model/{args.model_to_load}.pkl", "rb"))
    else:
        pipeline = None

    if args.type_of_run == "train":
        logger.info(f"the train is starting")
        if not isfile(f'data/{train_file.split(".")[0]}_w_lang.csv'):
            train = make_dataframe_and_add_languages(input_file = train_file)
        else:
            train = pd.read_csv(f'data/{train_file.split(".")[0]}_w_lang.csv')
        train_df_fix_category = naive_group_category(train, select_lang=args.select_lang)

        classifier = ClassifierModel(input_data=train_df_fix_category,
                                     tfidf_max_feat=500,
                                     mlflow_local=True,
                                     classifier_type='lgbm',
                                     use_data_under_balancer=True,
                                     vectorizer_type='tfidf',
                                     optimum_n_components=None,
                                     pipeline=pipeline,
                                     mlflow_local_url='mlruns')
        classifier.train(grid_search=False)

    if args.type_of_run == "predict":
        if not isfile(f'data/{final_test_file.split(".")[0]}_w_lang.csv'):
            test = make_dataframe_and_add_languages(input_file = final_test_file)
        else:
            test = pd.read_csv(f'data/{final_test_file.split(".")[0]}_w_lang.csv')

        if args.select_lang:
            test = test[test['language'] == args.select_lang]

        classifier = ClassifierModel(pipeline=pipeline)

        test['pred_category'] = classifier.predict(test, proba=False)

        with open('data/relabeling_dict.yml', 'r') as f:
            relabeling_dict = yaml.load(f)

        labeling_dict = dict(map(reversed, relabeling_dict.items()))

        test['pred_category'] = test['pred_category'].apply(lambda x: labeling_dict[x])

        test.to_csv(f'data/{final_test_file.split(".")[0]}_{args.select_lang}.csv', index=False)
