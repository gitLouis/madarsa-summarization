from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from bertopic.representation import MaximalMarginalRelevance

from hdbscan import HDBSCAN
from umap import UMAP
from pprint import pp

from preprocessing.sentence_splitting import explode_sentences
from modeling.LLM_queries import topic_info_to_keypoints
import pandas as pd

def build_topic_model(verbose=0):
    topic_size_ = 10
    umap_model = UMAP(n_components=16, n_neighbors=3, min_dist=0.0)
    hdbscan_model = HDBSCAN(min_cluster_size = topic_size_, gen_min_span_tree=True, prediction_data=True, min_samples=4)
    sentence_model = SentenceTransformer("imvladikon/sentence-transformers-alephbert")  # all-MiniLM-L6-v2
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)
    representation_model = MaximalMarginalRelevance(diversity=0.1)
    # topic_model = BERTopic(embedding_model=sentence_model, representation_model=representation_model)

    bert_config = {'language':"hebrew",
                'embedding_model':sentence_model,
                'top_n_words':20,
                'n_gram_range':(1, 4),
                'min_topic_size':topic_size_,
                'nr_topics':15,
                'low_memory':False,
                'calculate_probabilities':False,
                'umap_model':umap_model,
                'hdbscan_model':hdbscan_model,
                'ctfidf_model':ctfidf_model,
                'representation_model':representation_model
                }


    topic_model = BERTopic(**bert_config)
    if verbose>8:
        pp(topic_model.get_params())
    return topic_model

def plot_hierarchical_topics(topic_model, docs, save_to_file):
    hierarchical_topics = topic_model.hierarchical_topics(docs=docs)
    fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    fig.write_html(save_to_file.with_suffix('.html'))
    print(f'Saved to {save_to_file}')


def fit_transform_model_topic(col, verbose=0, hierarchical_plot_file=''):
    if verbose:
        print('='*80)
        print(f'{col.name} TOPIC MODELING')
        print('='*80)
    sentences = explode_sentences(col)
    topic_model = build_topic_model(verbose=verbose)
    topic_model.fit(sentences) # .fillna('NA')
    topics, prob = topic_model.transform(sentences)
    # representative_docs = topic_model.get_representative_docs()

    topic_info = topic_model.get_topic_info()
    topic_info['keywords'] = topic_info.apply(topic_info_to_keypoints, axis=1)
    topic_info.set_index('Topic', inplace=True)

    if hierarchical_plot_file:
        plot_hierarchical_topics(topic_model, docs=sentences, save_to_file=hierarchical_plot_file)

    sentences_with_topics = pd.DataFrame({'sentences':sentences, 
                                        'topic':topics})

    grouped_by_topics = sentences_with_topics.groupby('topic')['sentences'].agg([lambda x: list(set(x)),'size']).reset_index()
    grouped_by_topics.columns = ['topic','list', 'num_samples']
    grouped_by_topics['num_unique'] = grouped_by_topics['list'].apply(len)
    grouped_by_topics.set_index('topic', inplace=True)
    grouped_by_topics = grouped_by_topics.merge(topic_info, left_index=True, right_index=True)

    return grouped_by_topics, topic_model

