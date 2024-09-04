import os
import time
from pathlib import Path
import pandas as pd
import re
import pickle
import pytest
import nltk

from src.preprocessing.data_loader import basic_data_loader
from src.preprocessing.sentence_splitting import explode_sentences
from src.modeling.topic_modeling import fit_transform_model_topic
from src.modeling.LLM_queries import summarize_topic

def _run_full_cycle(num_rows_per_col):

    cwd = os.getcwd()
    root = Path(cwd).parent
    # src_path = root/'src' 
    # sys.path.insert(0, str(src_path))


    # prepare input/output folder paths

    data_folder = root/'data'
    assert data_folder.exists(), f'{data_folder=} is missing.'
    experiment_data_folder = data_folder/'questionnaire-data-july-1-2024'
    assert experiment_data_folder.exists(), f'{experiment_data_folder=} is missing.'

    print('Downloading nltk data might take a few minutes...why not take a coffee break.')
    nltk.download('punkt_tab', download_dir=root/'venv'/'nltk_data')

    experiement_name = f'experiment_{int(time.time())}'
    main_output_folder = experiment_data_folder/'output'
    main_output_folder.mkdir(parents=False, exist_ok=True)
    output_folder = main_output_folder/experiement_name
    output_folder.mkdir(parents=False, exist_ok=True)
    print(f'New output folder created:\t{str(output_folder)}')

    dfs, text_col_names = basic_data_loader(experiment_data_folder/'raw')


    def topic_model_and_summarize_column(col_: pd.Series, save_to_folder: str|Path, verbose=0):
        col_name = re.sub(r'\W+', ' ', col_.name)
        col_name.strip()
        hierarchical_plot_file = save_to_folder/col_name
        sentences_col = explode_sentences(col_)
        sentences_col = sentences_col[:300]
        grouped_by_topics, topic_model = fit_transform_model_topic(sentences_col, verbose=verbose, hierarchical_plot_file=hierarchical_plot_file)
        grouped_by_topics['summary'] = grouped_by_topics['list'].apply(lambda l : summarize_topic(col_.name, l, context_length=250, verbose=verbose))
        grouped_by_topics['summary_content'] = grouped_by_topics['summary'].apply(lambda x: x['message']['content'].replace('\n',''))
        return grouped_by_topics


    responses = dict()
    for qstr_key, df in dfs.items():
        print(qstr_key)
        qstr_responses = dict()
        for col_name in text_col_names[qstr_key]:
            col = df[col_name].dropna()
            col = col[:num_rows_per_col]
            col_responses = topic_model_and_summarize_column(col, output_folder, verbose=7)
            qstr_responses[col_name] = col_responses
            break
        responses[qstr_key] = qstr_responses
        break


    ### Save to pickle
    with open(output_folder/'responses.pkl', 'wb') as file:
        pickle.dump(responses, file)

    ###  Load from pickle
    # with open(output_folder/'responses.pkl', 'rb') as file:
    #     loaded_dfs = pickle.load(file)

    for qstr_key, v in responses.items():
        for question, summary in v.items():
            summary['summary_content'] = summary['summary'].apply(lambda x: x['message']['content'].replace('\n',''))
    print(summary)

def test_full_cycle():
    try:
        _run_full_cycle(num_rows_per_col = 100)
    except Exception as e:
        pytest.fail(f"Function raised an exception: {e}")
