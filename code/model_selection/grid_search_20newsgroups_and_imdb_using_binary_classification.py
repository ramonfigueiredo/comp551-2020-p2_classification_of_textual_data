from grid_search.run_grid_search import run_grid_search


def run_grid_search_20newsgroups_and_imdb_using_binary_classification():
    run_grid_search(save_logs_in_file=True, just_imdb_dataset=False, imdb_multi_class=False, save_json_with_best_parameters=True)
