# ID2214 Lab 1 - Predict air quality

Content of the notebook, unless mentioned here, remains unchanged from the original template. This is especially true for the non-lagged single day feature pipeline, which had very minimal changes.

## Air quality sensor
From [WAQI](https://waqi.info/), we picked a sensor with ID A65074 in [Lokuciewskiego, Warszawa, Poland](https://aqicn.org/station/@65074/), mainly due to its data consistency dating back to 5 years ago. The CSV file with historical data is located [here](../../data/Lokuciewskiego-Warszawa.csv).

## 3 day lagged feature
For ease of maintenance, we created a copy of each notebook for the lagged air quality feature. But apart from the changed mentioned below, the content of the notebooks remains the same as the single day feature.

For the lagged `pm25` values of 3 days (notebook 1), we create three new columns in the `air_quality_lagged` feature group - `lagged_1`, `lagged_2`, and `lagged_3` - by shifting the `pm25` column by 1, 2, and 3 days respectively. Since shifting, the lagged columns will have NaN values in initial 1-3 rows. Which we will drop before training the model.

```python
df_aq_lagged['lagged_1'] = df_aq_lagged['pm25'].shift(1)
```

During the feature pipeline (notebook 2), we modify the feature group as:
- Store `lagged_2` to `laggged_3`
- Store `lagged_1` to `laggged_2`
- Store `pm25` to `laggged_1`
- Store the new (current day's) air quality values in `pm25` column

For the feature view `air_quality_lagged_fv`, along with the `pm25` values, we also select the `lagged_1`, `lagged_2`, and `lagged_3` columns.

While the non-lagged `pm25` considered the mean temperature as the most important feature, the lagged `pm25` values could depend on the previous day's air quality values to produce better MSE and R2 scores (notebook 3).

For generating inferences from the model for the following week (notebook 4), we will not have lagged air quality values for the future days for any future day. So, we iteratively predict each day of the week by feeding the previous day's prediction as the lagged air quality value.

1. Day 1: Predict with day 0, -1 and -2. (all real values)
2. Day 2: Predict with day 1, 0 and -1. (1 being the results of step 1)
3. Day 3: Predict with day 2, 1 and 0. (2 and 1 being the results of step 2 and 1, respectively)

and so on.

## Miscellaneous
### Test-train split
Since we did not get to change the example notebook, the analyses are based on splitting the dataset on 2024-10-15. For our case, this effectively becomes a 98-2 split.