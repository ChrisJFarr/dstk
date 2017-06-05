# Parallel processing normalizatin  https://blog.dominodatalab.com/simple-parallelization/
from joblib import Parallel, delayed
import multiprocessing
from sklearn.model_selection import KFold
from normalization import normalize_corpus


def process(text):
    num_cores = multiprocessing.cpu_count()
    # Create splits for parallel jobs
    folds = KFold(n_splits=num_cores, random_state=0, shuffle=False)
    all_folds = []
    for fold in folds.split(text):
        all_folds.append(fold[1])

    results_par = Parallel(n_jobs=num_cores)(delayed(normalize_corpus)([text[j] for j in i]) for i in all_folds)
    norm_text = [item for sublist in results_par for item in sublist]

    return norm_text
