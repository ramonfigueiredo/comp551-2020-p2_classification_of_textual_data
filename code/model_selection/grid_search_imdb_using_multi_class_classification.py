from grid_search.run_grid_search import run_grid_search


def run_grid_search_imdb_using_multi_class_classification():
    run_grid_search(save_logs_in_file=True, just_imdb_dataset=True, imdb_multi_class=True, save_json_with_best_parameters=True)
