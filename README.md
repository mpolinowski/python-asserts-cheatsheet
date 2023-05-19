# Python Asserts Cheat Sheet


## Datasets

__Validating and Verifying Data__


* [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

Yellow and green taxi trip records include fields capturing pick-up and drop-off dates/times, pick-up and drop-off locations, trip distances, itemized fares, rate types, payment types, and driver-reported passenger counts. The data used in the attached datasets were collected and provided to the NYC Taxi and Limousine Commission (TLC) by technology providers authorized under the Taxicab & Livery Passenger Enhancement Programs (TPEP/LPEP). The trip data was not created by the TLC, and TLC makes no representations as to the accuracy of these data.


<!-- TOC -->

- [Python Asserts Cheat Sheet](#python-asserts-cheat-sheet)
  - [Datasets](#datasets)
  - [Introduction to Asserts](#introduction-to-asserts)
    - [Asserts in Python](#asserts-in-python)
    - [Asserts in Pandas](#asserts-in-pandas)
      - [Indices](#indices)
      - [Series](#series)
      - [DataFrames](#dataframes)
    - [Asserts in Numpy](#asserts-in-numpy)
  - [Assert-based Testing](#assert-based-testing)
    - [Quantitative Tests](#quantitative-tests)
    - [Logical Tests](#logical-tests)

<!-- /TOC -->

```python
!wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet -P dataset
```

```python
import csv
import math
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
```

```python
yellow_tripdata_df = pd.read_parquet(
    'dataset/yellow_tripdata_2023-01.parquet'
)

yellow_tripdata_df.to_csv('dataset/yellow_tripdata_2023-01.csv')
```

```python
yellow_tripdata_df = pd.read_csv(
    'dataset/yellow_tripdata_2023-01.csv',
    parse_dates=['tpep_pickup_datetime','tpep_dropoff_datetime'],
    nrows=1000
)
yellow_tripdata_df.head(5)
```

```python
# https://www.kaggle.com/datasets/neomatrix369/nyc-taxi-trip-duration-extended
trip_ext_df = pd.read_csv('dataset/nyc_trip_duration_extended.csv')
trip_ext_df.head(5)
```

## Introduction to Asserts

### Asserts in Python

```python
# simple assert
x = 'five'
assert x == 5
# AssertionError
```

```python
list = [6,2,3,4,5]
assert all(list[i] <= list[i+1] for i in range(len(list)-1))
# AssertionError
```

```python
def add(a,b):
    return a + b

assert add(2,3) < 5
AssertionError
```

```python
trip_ext_df.columns

# Index(['name', 'district', 'neighbourhood', 'latitude', 'longitude',
#        'geonumber'],
#       dtype='object')
```

```python
with open('dataset/nyc_trip_duration_extended.csv') as f:
    reader = csv.DictReader(f)
    
    expected_columns = ['name', 'district', 'neighbourhood', 'latitude', 'longitude',
       'geonumber', 'missing_column']
    
    assert reader.fieldnames == expected_columns, f"Expected columns: {expected_columns}, but got {reader.fieldnames}"
    
# AssertionError:
# Expected columns: ['name', 'district', 'neighbourhood', 'latitude', 'longitude', 'geonumber', 'missing_column'],
# but got ['name', 'district', 'neighbourhood', 'latitude', 'longitude', 'geonumber']

```

```python
with open('dataset/yellow_tripdata_2023-01.csv') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        # check passenger count is positive int
        assert float(row['passenger_count']) > 0., f"ERROR :: Invalid Passenger Count: {row['passenger_count']}"
        
# AssertionError: ERROR :: Invalid Passenger Count: 0.0
```

```python
# how many trips were without passengers
trips_without_passengers = 0

with open('dataset/yellow_tripdata_2023-01.csv') as f:
    next(f) # # skip header
    
    for line in f:
        values = line.strip().split(',')
        trip_id = values[0]
        passenger_count = values[4]
        if passenger_count == '0.0':
            trips_without_passengers += 1
            
print(f'Trips without passengers: {trips_without_passengers}')
# Trips without passengers: 51164
```

```python
perct_zero_trips = len(zero_trips) * 100 / len(yellow_tripdata_df)
print("%.2f" % perct_zero_trips + ' %')
# 1.67 %
```

### Asserts in Pandas

#### Indices

```python
index1 = pd.Index([1,2,3])
index2 = pd.Index([1,2,'three'])

pdt.assert_index_equal(index1, index2)

# Index classes are different
# [left]:  Int64Index([1, 2, 3], dtype='int64')
# [right]: Index([1, 2, 'three'], dtype='object')
```

```python
index1 = pd.Index([1,2,3])
index2 = pd.Index([3,2,1])

pdt.assert_index_equal(index1, index2, check_order=True)
# Index values are different (66.66667 %)
# [left]:  Int64Index([1, 2, 3], dtype='int64')
# [right]: Int64Index([3, 2, 1], dtype='int64')
```

```python
index1 = pd.Index([1.0,2.0,3.0])
index2 = pd.Index([1.0,2.0,3.1])

pdt.assert_index_equal(index1, index2, check_exact=False, atol=0.1)

# Index values are different (33.33333 %)
# [left]:  Float64Index([1.0, 2.0, 3.0], dtype='float64')
# [right]: Float64Index([1.0, 2.0, 3.1], dtype='float64')
```

```python
index1 = pd.Index(yellow_tripdata_df['tpep_pickup_datetime'].dt.date)
index1[:3]
# Index([2023-01-01, 2023-01-01, 2023-01-01], dtype='object', name='tpep_pickup_datetime')
```

```python
index2 = pd.Index(yellow_tripdata_df['tpep_dropoff_datetime'].dt.date)
index2[:3]
# Index([2023-01-01, 2023-01-01, 2023-01-01], dtype='object', name='tpep_dropoff_datetime')
```

```python
pdt.assert_index_equal(index1, index2, check_exact=True, check_names=False)

# Index values are different (0.3 %)
# [left]:  Index([2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01,
#        2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01,
#        ...
#        2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01,
#        2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01],
#       dtype='object', name='tpep_pickup_datetime', length=1000)
# [right]: Index([2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01,
#        2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01,
#        ...
#        2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01,
#        2023-01-01, 2023-01-01, 2023-01-01, 2023-01-01],
#       dtype='object', name='tpep_dropoff_datetime', length=1000)
```

```python
# show difference
index_diff = yellow_tripdata_df[
    index1 != index2
][
    ['tpep_pickup_datetime','tpep_dropoff_datetime']
]

index_diff.head()

#     tpep_pickup_datetime tpep_dropoff_datetime
# 383  2023-01-01 00:36:07   2023-01-02 00:17:13
# 567  2022-12-31 23:59:37   2023-01-01 00:07:28
# 761  2022-12-31 23:58:27   2023-01-01 00:02:21
```

#### Series

```python
s1 = pd.Series([1,2,3], name='series1')
s2 = pd.Series([1,2,3], name='series2')

pdt.assert_series_equal(s1,s2,check_names=True)

# Attribute "name" are different
# [left]:  series1
# [right]: series2
```

```python
s1 = pd.Series([1,2,3], name='series1')
s2 = pd.Series(['1','2','3'], name='series2')

pdt.assert_series_equal(s1,s2,check_names=False, check_dtype=False)

# Series values are different (100.0 %)
# [index]: [0, 1, 2]
# [left]:  [1, 2, 3]
# [right]: [1, 2, 3]
```

```python
pickup_series = yellow_tripdata_df['tpep_pickup_datetime'].dt.date
pickup_series.head(2)

# 0    2023-01-01
# 1    2023-01-01
# Name: tpep_pickup_datetime, dtype: object
```

```python
dropoff_series = yellow_tripdata_df['tpep_dropoff_datetime'].dt.date
pickup_series.head(2)

# 0    2023-01-01
# 1    2023-01-01
# Name: tpep_pickup_datetime, dtype: object
```

```python
pdt.assert_series_equal(pickup_series, dropoff_series, check_exact=True, check_names=False)

# AssertionError: Series are different
# Series values are different (0.3 %)
```

```python
# drop values that don't fit
index_diff = pickup_series.index[pickup_series != dropoff_series]
yellow_tripdata_df_drop = yellow_tripdata_df.drop(index_diff).reset_index(drop=True)
# rebuild series
pickup_series = yellow_tripdata_df_drop['tpep_pickup_datetime'].dt.date
dropoff_series = yellow_tripdata_df_drop['tpep_dropoff_datetime'].dt.date
# re-check - this times it works
pdt.assert_series_equal(pickup_series, dropoff_series, check_exact=True, check_names=False)
```

#### DataFrames

```python
df1 = pd.DataFrame({'A': [1,2,3], 'B': [3,2,1]})
df2 = pd.DataFrame({'B': [3,2,1], 'A': [1,2,3]})

pdt.assert_frame_equal(df1,df2,check_like=False)

# DataFrame.columns values are different (100.0 %)
# [left]:  Index(['A', 'B'], dtype='object')
# [right]: Index(['B', 'A'], dtype='object')
```

```python
pickup_df = yellow_tripdata_df.copy()
dropoff_df = yellow_tripdata_df.copy()

pickup_df['date'] = yellow_tripdata_df['tpep_pickup_datetime'].dt.date
dropoff_df['date'] = yellow_tripdata_df['tpep_dropoff_datetime'].dt.date

pdt.assert_frame_equal(
    pickup_df[['date']],
    dropoff_df[['date']],
    check_exact=True,
    check_names=False
)

# AssertionError: DataFrame.iloc[:, 0] (column name="date") are different
# DataFrame.iloc[:, 0] (column name="date") values are different (0.3 %)
```

```python
# get index of mismatched rows
index_diff = pickup_df.index[
    pickup_df['date'].ne(dropoff_df['date'])
]

index_diff
# Int64Index([383, 567, 761], dtype='int64')
```

```python
# drop rows at those indices
pickup_df_drop = pickup_df.drop(index_diff)
dropoff_df_drop = dropoff_df.drop(index_diff)

# verify that assert now works
pdt.assert_frame_equal(
    pickup_df_drop[['date']],
    dropoff_df_drop[['date']],
    check_exact=True,
    check_names=False
)
```

### Asserts in Numpy

```python
a = np.array([1,2,3])
b = np.array([3,2,1])

npt.assert_array_equal(a,b)

# AssertionError: Arrays are not equal
# Mismatched elements: 2 / 3 (66.7%)
# Max absolute difference: 2
# Max relative difference: 2.
#  x: array([1, 2, 3])
#  y: array([3, 2, 1])
```

```python
string1 = 'string'
string2 = 'STRING'

npt.assert_string_equal(string1,string2)
```

```python
a = np.array([1,2,3])
b = np.array([0.98,2.02,2.98])

npt.assert_allclose(a,b,atol=0.01)

# AssertionError: Not equal to tolerance rtol=1e-07, atol=0.01
# Mismatched elements: 3 / 3 (100%)
# Max absolute difference: 0.02
# Max relative difference: 0.02040816
# x: array([1, 2, 3])
# y: array([0.98, 2.02, 2.98])
```

```python
a = np.array([1,2,3])
b = np.array([2,3,4])

npt.assert_array_less(a,b)
```

```python
npt.assert_array_less(b,a)

# AssertionError: Arrays are not less-ordered
# Mismatched elements: 3 / 3 (100%)
# Max absolute difference: 1
# Max relative difference: 1.
#  x: array([2, 3, 4])
#  y: array([1, 2, 3])
```

## Assert-based Testing

### Quantitative Tests

```python
def test_for_missing_data(df):
    # count all missing values and assert number to be zero
    assert df.isnull().sum().sum() == 0, 'ERROR :: DataFrame contains missing data!'
    return True
```

```python
assert test_for_missing_data(trip_ext_df)
# True
```

```python
def test_non_numerical_data_types(df, columns):
    for col in columns:
        assert df[col].dtype == 'int64' or df[col].dtype =='float64', f'ERROR :: {col} has a non-numerical dType'
    return True
```

```python
test_columns = ['neighbourhood','latitude','longitude','geonumber']
assert test_non_numerical_data_types(trip_ext_df, trip_ext_df[test_columns])
# AssertionError: ERROR :: neighbourhood has a non-numerical dType
```

```python
def test_for_out_of_range(df, columns):
    for col in columns:
        assert df[col].dtype == 'int64' or df[col].dtype == 'float64', f'ERROR :: {col} has a non-numerical dType'
        assert df[col].max() <= math.inf, f'ERROR :: {col} contains infinite values'
        assert df[col].min() >= -math.inf, f'ERROR :: {col} contains infinite values'
        assert not np.isnan(df[col]).any(), f'ERROR :: {col} contains NaN values'
        assert not np.isinf(df[col]).any(), f'ERROR :: {col} contains infinite values'
    return True
```

```python
test_columns = ['latitude','longitude','geonumber']
assert test_non_numerical_data_types(trip_ext_df, trip_ext_df[test_columns])
# True
```

### Logical Tests

```python
def test_for_logical_errors(df):
    # all dropoffs AFTER pickups
    assert all(df['tpep_dropoff_datetime'] > df['tpep_pickup_datetime']), 'ERROR :: Drop-off time before pickup'
    # no negative trip distances
    assert (df['trip_distance'] >= 0).all(), 'ERROR :: Negative trip distances'
    # no negative passenger count
    assert (df['passenger_count'] >= 0).all(), 'ERROR :: Negative passenger count'
    
    return True
```

```python
assert test_for_logical_errors(yellow_tripdata_df)
# True
```