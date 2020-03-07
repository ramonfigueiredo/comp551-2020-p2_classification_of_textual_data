from grid_search.run_grid_search import run_grid_search


if __name__ == '__main__':
    run_grid_search(save_logs_in_file=True, just_imdb_dataset=True, imdb_multi_class=True, save_json_with_best_parameters=True)
