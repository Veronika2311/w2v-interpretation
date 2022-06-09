# -*- coding: utf-8 -*-
"""
Created on Sat May 14 18:02:18 2022

@author: veron
"""

import re
import gensim
import logging
import nltk.data
import pandas as pd
from nltk.corpus import stopwords
from gensim.models import word2vec

from tqdm import tqdm
import numpy as np
import wget
import zipfile
import random

import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from umap import UMAP
from sklearn.cluster import KMeans

import pymorphy2
morph = pymorphy2.MorphAnalyzer()


def get_matrix(all_nouns, model):
    vectors_of_words = np.zeros((len(all_nouns), model.vector_size))
    for i, word in enumerate(all_nouns):
        try:
            vectors_of_words[i] = model[word]
        except:
            print(word)
    return vectors_of_words


def lsa_matrix(vectors_of_words, n_components, n_iter=100):
    lsa_obj = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=42)
    lsa_data = lsa_obj.fit_transform(vectors_of_words)
    return lsa_data


def sort_results(lsa_data, all_nouns):    
    sorted_scores_indx = np.argsort(lsa_data, axis=0)[::-1]
    result = np.array(all_nouns)[sorted_scores_indx.ravel()]
    result_nums = np.array(lsa_data)[sorted_scores_indx.ravel()]
    return result, result_nums


def sort_results2(lsa_data, all_nouns, all_lsa):    
    sorted_scores_indx = np.argsort(lsa_data, axis=0)[::-1]
    result = np.array(all_nouns)[sorted_scores_indx.ravel()]
    result_nums = np.array(all_lsa)[sorted_scores_indx.ravel()]
    return result, result_nums


def get_n_iterations(all_nouns, model, iterations):
    dict_iters = {'0': [all_nouns]}
    for i in range(iterations):
        iter_name = str(i + 1)
        dict_iters[iter_name] = []
        for el in dict_iters[str(i)]:
            if len(el) > 50:
                first_matrix = get_matrix(el, model)
                first_lsa = lsa_matrix(first_matrix, i + 1, 200)
                first_result, first_result_num = sort_results([v[0] for v in first_lsa], el)
                part_of_list = len(first_result) // 5
                dict_iters[iter_name].append(first_result[:part_of_list])
                dict_iters[iter_name].append(first_result[2*part_of_list:])
            else:
                dict_iters[iter_name].append(el)
                
    return dict_iters


def clustering(vectors_of_words,  eps=3, min_samples=1000):
    #clustering_data = DBSCAN(eps=eps, min_samples=first_matrix.shape[0]/5).fit(vectors_of_words)
    clustering_data = sklearn.cluster.KMeans(n_clusters=vectors_of_words.shape[0]//eps).fit_predict(vectors_of_words)
    return clustering_data


def separate_words(first_matrix, first_lsa, all_nouns, clust_size):
    first_clustering = clustering(first_matrix, clust_size)
    m, d = np.mean(first_lsa), np.std(first_lsa)

    processed = [-1 for i in range(max(first_clustering)+1)]
    distr = [[0, 0, 0] for i in range(max(first_clustering)+1)]

    for val, cl in zip(first_lsa, first_clustering):
        if val < m - d:
            distr[cl][0] += 1
        elif val > m + d:
            distr[cl][2] += 1
        else:
            distr[cl][1] += 1

    words = [[], [], []]
    for word, cl in zip(all_nouns, first_clustering):
        pos = np.argmax(distr[cl])
        words[pos].append(word)

    return words


def separate_words2(first_matrix, first_lsa, clust_size):
    #first_clustering = clustering(first_matrix, clust_size)
    m, d = np.mean(first_lsa), np.std(first_lsa)

#     processed = [-1 for i in range(max(first_clustering)+1)]
#     distr = [[0, 0, 0] for i in range(max(first_clustering)+1)]

    words = [[], [], []]
    for word, val in zip(first_matrix, first_lsa):
        if val > m + 1/2*d:
            words[0].append(word)
        elif val < m - 1/2*d:
            words[2].append(word)
        else:
            words[1].append(word)

    return words


def get_n_iterations2(all_nouns, model, iterations):
    
    clust_size = {'0': 100, '1': 30, '2': 30, '3': 10, '4':10, '5':10, '6':10, '7':10, '8':10, '9':10, '10':10}
    dict_iters = {'0': [all_nouns]}
    dict_iters_num = {'0': []}
    #print('aaaaaaaaa')
    for i in range(iterations):
        print(i)
        iter_name = str(i + 1)
        dict_iters[iter_name] = []
        dict_iters_num[iter_name] = []
        for one_nouns_list in dict_iters[str(i)]:
            if len(one_nouns_list) > 20:
                first_matrix = get_matrix(one_nouns_list, model)
                first_lsa = lsa_matrix(first_matrix, i + 1, 200)
                first_result, first_result_num = sort_results([v[-1] for v in first_lsa], one_nouns_list)
                print(first_result[:50], first_result_num[:50])
                words = separate_words2(first_result, first_result_num, clust_size[str(i)])
                print(words[0][:20])
                dict_iters[iter_name].extend(words)
            else:
                dict_iters[iter_name].append(one_nouns_list)
            
    return dict_iters #, dict_iters_num


#without center, 3 parts
def get_n_iterations3(all_nouns, model, iterations):
    dict_iters = {'0': [all_nouns]}
    for i in range(iterations):
        print(i)
        iter_name = str(i + 1)
        dict_iters[iter_name] = []
        for el in dict_iters[str(i)]:
            first_matrix = get_matrix(el, model)
            first_lsa = lsa_matrix(first_matrix, i + 1, 200)
            first_result, first_result_num = sort_results([v[-1] for v in first_lsa], el)
            print(first_result[:50], first_result_num[:50])
            half_of_list = len(first_result) // 3.0
            dict_iters[iter_name].append(first_result[:half_of_list])
            dict_iters[iter_name].append(first_result[2*half_of_list:])
    return dict_iters


#with center, 3 parts
def get_n_iterations4(all_nouns, model, iterations):
    dict_iters = {'0': [all_nouns]}
    for i in range(iterations):
        print(i)
        iter_name = str(i + 1)
        dict_iters[iter_name] = []
        for el in dict_iters[str(i)]:
            first_matrix = get_matrix(el, model)
            first_lsa = lsa_matrix(first_matrix, i + 1, 200)
            first_result, first_result_num = sort_results([v[-1] for v in first_lsa], el)
            half_of_list = len(first_result) // 3
            dict_iters[iter_name].append(first_result[:half_of_list])
            dict_iters[iter_name].append(first_result[half_of_list:2*half_of_list])
            dict_iters[iter_name].append(first_result[2*half_of_list:])
    return dict_iters


#with center, 2 parts
def get_n_iterations5(all_nouns, model, iterations):
    dict_iters = {'0': [all_nouns]}
    for i in range(iterations):
        print(i)
        iter_name = str(i + 1)
        dict_iters[iter_name] = []
        for el in dict_iters[str(i)]:
            first_matrix = get_matrix(el, model)
            first_lsa = lsa_matrix(first_matrix, i + 1, 200)
            first_result, first_result_num = sort_results([v[-1] for v in first_lsa], el)
            half_of_list = len(first_result) // 2
            dict_iters[iter_name].append(first_result[:half_of_list])
            dict_iters[iter_name].append(first_result[half_of_list:])
    return dict_iters





def get_all_nouns(model):
    all_nouns = []
    if len(model.index_to_key[0].split('_')) == 2:
        for w in tqdm(model.index_to_key):
            wordform, pref = w.split('_')
            if pref == 'NOUN' and len(wordform) > 2:
                first_res_methods = morph.parse(wordform)[0].methods_stack
                if str(first_res_methods[0][0]) == 'DictionaryAnalyzer()' and len(first_res_methods) == 1:
                    all_nouns.append(w)
    else:
        for w in tqdm(model.index_to_key):
            parse_of_word = morph.parse(w)
            first_res_methods = parse_of_word[0].methods_stack
            if parse_of_word[0].tag.POS == 'NOUN' and len(w) > 2:
                if str(first_res_methods[0][0]) == 'DictionaryAnalyzer()' and len(first_res_methods) == 1:
                    all_nouns.append(w) 
            
    return all_nouns


import os
import re
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
import math



def preproc(text):
    preproc_text = []
    text = re.split('[ -,.\n]', text)
    for word in text:
        if len(word) > 2 and re.match('^[А-Яа-яЁё]+$', word):
            preproc_text.append(word.lower())
    return set(preproc_text)

def check_clusters(clusts, list_of_words):
    res_list = []
    for clust in clusts:
        clust_list = []
        for word in clust:
            if word.split('_')[0] in list_of_words:
                clust_list.append(1)
            else:
                clust_list.append(-1)
        res_list.append(clust_list)

    return res_list


def get_dict_histplot(clusts):
    path_to_dicts = './dicts'
    dict_od_dicts = {}

    for path, dirs, files in os.walk(path_to_dicts):
        for file in tqdm(files):
            with open(os.path.join(path, file), 'r', encoding='utf8') as f:
                words = preproc(f.read())
            dict_od_dicts[file] = [Counter(i) for i in check_clusters(clusts, words)]

    all_terms_words = {}
    
    for key, value in dict_od_dicts.items():
        all_terms_words[key] = [math.log1p(i[1]  / (1 + sum([el[1] for el in value]))) for i in value]
        #all_terms_array.append([i[1]  / (1 + sum([el[1] for el in value])) for i in value])
        #all_terms_array.append([i[1] for i in value])

    ordered_list = ['animacy.txt', 'archaic_words_one_word_terms.txt',  'russian_names_men.txt',
                    'russian_names_women.txt', 'russian_rivers.txt', 'russian_cities.txt', 'anatomy_one_word_terms.txt',
                    'plants_one_word_terms.txt',   'minerals.txt', 'geo.txt', 'geo_epoche.txt', 'knife.txt',
                    'fortification_terms.txt', 'rangs.txt',
                    'architecture.txt', 'arts.txt',
                    'informatics.txt',
                    'philology.txt', 'philosophy.txt', 
                    'politics.txt', 'professions.txt', 'russian_vacancy.txt']

    
    plt.figure(figsize=(15,8))
    sns.heatmap(pd.DataFrame([all_terms_words[i] for i in ordered_list], ordered_list))
