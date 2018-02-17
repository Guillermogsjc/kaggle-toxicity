from nltk.metrics.distance import  edit_distance
from itertools import  permutations
import numpy as np
import pandas as pd
from functools import partial
import multiprocessing as mp
import unicodedata


def accent_remove(s):
    return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))

def lev_no_case_sens(a, b):
    a = a.lower()
    b = b.lower()
    dist = edit_distance(a, b)
    return dist

def bow_dist(sec_str, main_str, case_sens=False):
    """ Calcula la mínima distancia entre las posibles reordenaciones.
    Normaliza por la longitud de la string principal, NO TIENE EN CUENTA TILDES"""
    main_str = accent_remove(main_str).strip().split()
    sec_str = accent_remove(sec_str).strip().split()
    n = len(main_str)
    m = len(sec_str)
    total_chars = np.sum([len(k) for k in main_str])
    n_perms  = int(np.min([n, m]))

    if n>m:
        aux = main_str.copy()
        main_str = sec_str.copy()
        sec_str = aux


    dist = np.inf

    if case_sens:
        for perm in list(permutations(sec_str, n_perms)):
            d_aux = np.sum([edit_distance(main_str[k], perm[k]) for k in range(n_perms)])
            if d_aux < dist: dist = d_aux
    else:
        for perm in list(permutations(sec_str, n_perms)):
            d_aux = np.sum([lev_no_case_sens(main_str[k], perm[k]) for k in range(n_perms)])
            if d_aux < dist: dist = d_aux

    return dist / total_chars

def dist_string_to_series(main_str, sec_series , case_sens_=False):

    f_dist = partial(bow_dist, main_str=main_str, case_sens=case_sens_)
    val = sec_series.map(f_dist).min()

    return val

def dist_series_to_series(main_series, sec_series , case_sens_=False):

    index_ = main_series.index
    dist_series = pd.Series(np.nan, index=index_)

    for ind in index_:
        main_str_  = main_series.loc[ind]
        dist_series.loc[ind] = dist_string_to_series(main_str_, sec_series, case_sens_=False)

    return dist_series

def dist_text_to_series(main_text, sec_series , case_sens_=False):
    
    main_series = pd.Series(main_text.split())
    index_ = main_series.index
    dist_series = pd.Series(np.nan, index=index_)

    for ind in index_:
        main_str_  = main_series.loc[ind]
        dist_series.loc[ind] = dist_string_to_series(main_str_, sec_series, case_sens_=False)

    return dist_series



def dist_series_to_series_paralell(main_series, sec_series, case_sens_=False):

    main_series_red = main_series.drop_duplicates()
    sec_series = sec_series.drop_duplicates()

    with mp.Pool(mp.cpu_count() - 2) as workers:
        f_dist = partial(dist_string_to_series, sec_series=sec_series, case_sens_=case_sens_)
        args = list(main_series_red)
        dist_dict = pd.Series(list(workers.map(f_dist, args)), index=main_series_red.values).to_dict()

    dist_series = main_series.map(lambda x: dist_dict[x])

    return dist_series


if __name__=='__main__':
    a = ' Herbert Karagan'
    b = 'Karajan hervert wisloW  '
    # su distancia es 2

    q = 'hervert'
    w = 'erberth'
    e = 'Hervert'
    # su distancia es 3

    y = ' qW rJRs Pin'
    w = 'PiN qw qweqwe lñlqw svkf RHqT'
    j = 'Qw RjrS pInqw'

    serie1 = pd.Series(['qwe we', 'Ana Palacios', 'pedro biescas'])
    serie2 = pd.Series([' we qweR', ' Palacios ewAna ', ' biescas pedro'])


    print(edit_distance(q, w))
    print(edit_distance(q, e))
    print(lev_no_case_sens(q, e))
    print(bow_dist(a, b))
    # print(list(permutations('123', 2)))
    print(bow_dist(y, w))
    print(bow_dist(y, w, case_sens=True))
    print(bow_dist(y, j))
    print(bow_dist(y, j, case_sens=True))

    print(dist_string_to_series('pedro viehscas', serie2))
    print(dist_series_to_series(serie1, serie2))
    print(dist_series_to_series_paralell(serie1, serie2))
