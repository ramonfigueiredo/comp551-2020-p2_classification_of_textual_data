## Best results

| Best Algorithm | Dataset					    | Test accuracy with best epoch | Baseline test accuracy |
| -------------- | ---------------------------- | ----------------------------- | ---------------------- |
| KERAS DL 1     | 20 NEWS GROUPS (multi-class) | 96.69	   					    | 69.10 %                |
| KERAS DL 1 	 | IMDB REVIEWS (binary)	    | 88.36						    | 89.70 %                |
| KERAS DL 1	 | IMDB REVIEWS (multi-class)   | 89.10						    | ---                    |


Overall equation:
```
np.mean(np.concatenate([20_news_pred,IMDb_pred])==np.concatenate([twenty_test.target,imdb_test_target]))
```

* Overall baseline calculation: 0.8493 (84.93%)

```
(7532 * 0.691 + 25000 * 0.897) / (7532 + 25000) = 0.8493 (84.93%)
```

* Overal calculation for our case: 0.9029 (90.29%)

```
(7532 * 0.9669 + 25000 * 0.8836) / (7532 + 25000) = 0.9029 (90.29%)
```