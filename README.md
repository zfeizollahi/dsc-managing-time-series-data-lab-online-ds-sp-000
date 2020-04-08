
# Managing Time Series Data - Lab

## Introduction

In the previous lesson, you learned that time series data are everywhere and working with time series data is an important skill for data scientists!

In this lab, you'll practice your previously learned techniques to import, clean, and manipulate time series data.

The lab will cover how to perform time series analysis while working with large datasets. The dataset can be memory intensive so your computer will need at least 2GB of memory to perform some of the calculations.


## Objectives

You will be able to:

- Load time series data using Pandas and perform time series indexing 
- Perform data cleaning operation on time series data 
- Change the granularity of a time series 


## Let's get started!

Import the following libraries: 

* `pandas`, using the alias `pd` 
* `pandas.tseries` 
* `matplotlib.pyplot`, using the alias `plt` 
* `statsmodels.api`, using the alias `sm`


```python
# Load required libraries
import pandas as pd
import pandas.tseries
import matplotlib.pyplot as plt
import statsmodels.api as sm
```

## Loading time series data
The `statsModels` library comes bundled with built-in datasets for experimentation and practice. A detailed description of these datasets can be found [here](http://www.statsmodels.org/dev/datasets/index.html). Using `statsModels`, the time series datasets can be loaded straight into memory. 

In this lab, we'll use the **Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.**, containing CO2 samples from March 1958 to December 2001. Further details on this dataset are available [here](http://www.statsmodels.org/dev/datasets/generated/co2.html).

In the following cell: 

- We loaded the `co2` dataset using the `.load()` method 
- Converted this into a pandas DataFrame 
- Renamed the columns 
- Set the `'date'` column as index 


```python
# Load the 'co2' dataset from sm.datasets
data_set = sm.datasets.co2.load()

# load in the data_set into pandas data_frame
CO2 = pd.DataFrame(data=data_set['data'])
CO2.rename(columns={'index': 'date'}, inplace=True)

# set index to date column
CO2.set_index('date', inplace=True)

CO2.head()
```

    /opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/statsmodels/datasets/utils.py:344: FutureWarning: load will return datasets containing pandas DataFrames and Series in the Future.  To suppress this message, specify as_pandas=False
      FutureWarning)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>co2</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1958-03-29</td>
      <td>316.1</td>
    </tr>
    <tr>
      <td>1958-04-05</td>
      <td>317.3</td>
    </tr>
    <tr>
      <td>1958-04-12</td>
      <td>317.6</td>
    </tr>
    <tr>
      <td>1958-04-19</td>
      <td>317.5</td>
    </tr>
    <tr>
      <td>1958-04-26</td>
      <td>316.4</td>
    </tr>
  </tbody>
</table>
</div>



Let's check the data type of `CO2` and also print the first 15 entries of `CO2` as our first exploratory step.


```python
# Print the data type of CO2 


# Print the first 15 rows of CO2
CO2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 2284 entries, 1958-03-29 to 2001-12-29
    Data columns (total 1 columns):
    co2    2225 non-null float64
    dtypes: float64(1)
    memory usage: 35.7 KB



```python
CO2.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>co2</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1958-03-29</td>
      <td>316.1</td>
    </tr>
    <tr>
      <td>1958-04-05</td>
      <td>317.3</td>
    </tr>
    <tr>
      <td>1958-04-12</td>
      <td>317.6</td>
    </tr>
    <tr>
      <td>1958-04-19</td>
      <td>317.5</td>
    </tr>
    <tr>
      <td>1958-04-26</td>
      <td>316.4</td>
    </tr>
    <tr>
      <td>1958-05-03</td>
      <td>316.9</td>
    </tr>
    <tr>
      <td>1958-05-10</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1958-05-17</td>
      <td>317.5</td>
    </tr>
    <tr>
      <td>1958-05-24</td>
      <td>317.9</td>
    </tr>
    <tr>
      <td>1958-05-31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1958-06-07</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1958-06-14</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1958-06-21</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1958-06-28</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1958-07-05</td>
      <td>315.8</td>
    </tr>
  </tbody>
</table>
</div>



With all the required packages imported and the `CO2` dataset as a Dataframe ready to go, we can move on to indexing our data.

## Date Indexing

While working with time series data in Python, having dates (or datetimes) in the index can be very helpful, especially if they are of `DatetimeIndex` type. Further details can be found [here](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Timestamp.html).

Print the `.index` attribute of the `CO2` DataFrame: 


```python
# Confirm that date values are used for indexing purpose in the CO2 dataset 
CO2.index
```




    DatetimeIndex(['1958-03-29', '1958-04-05', '1958-04-12', '1958-04-19',
                   '1958-04-26', '1958-05-03', '1958-05-10', '1958-05-17',
                   '1958-05-24', '1958-05-31',
                   ...
                   '2001-10-27', '2001-11-03', '2001-11-10', '2001-11-17',
                   '2001-11-24', '2001-12-01', '2001-12-08', '2001-12-15',
                   '2001-12-22', '2001-12-29'],
                  dtype='datetime64[ns]', name='date', length=2284, freq=None)



The output above shows that our dataset clearly fulfills the indexing requirements. Look at the last line:


> **dtype='datetime64[ns]', length=2284, freq='W-SAT'**


* `dtype=datetime[ns]` field confirms that the index is made of timestamp objects.
* `length=2284` shows the total number of entries in our time series data. 

## Resampling

Remember that depending on the nature of analytical question, the resolution of timestamps can also be changed to other frequencies. For this dataset we can resample to monthly CO2 consumption values. This can be done by using the `.resample()` method as seen in the earlier lesson. 

* Group the data into buckets representing 1 month using `.resample()` method 
* Call the `.mean()` method on each group (i.e. get monthly average) 
* Combine the result as one row per monthly group 


```python
# Group the time series into monthly buckets
CO2_monthly = CO2.resample('MS')

# Take the mean of each group 
CO2_monthly_mean = CO2_monthly.mean()

# Get the first 10 elements of resulting time series
CO2_monthly_mean.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>co2</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1958-03-01</td>
      <td>316.100000</td>
    </tr>
    <tr>
      <td>1958-04-01</td>
      <td>317.200000</td>
    </tr>
    <tr>
      <td>1958-05-01</td>
      <td>317.433333</td>
    </tr>
    <tr>
      <td>1958-06-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1958-07-01</td>
      <td>315.625000</td>
    </tr>
    <tr>
      <td>1958-08-01</td>
      <td>314.950000</td>
    </tr>
    <tr>
      <td>1958-09-01</td>
      <td>313.500000</td>
    </tr>
    <tr>
      <td>1958-10-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1958-11-01</td>
      <td>313.425000</td>
    </tr>
    <tr>
      <td>1958-12-01</td>
      <td>314.700000</td>
    </tr>
  </tbody>
</table>
</div>




```python
CO2_monthly_mean.index
```




    DatetimeIndex(['1958-03-01', '1958-04-01', '1958-05-01', '1958-06-01',
                   '1958-07-01', '1958-08-01', '1958-09-01', '1958-10-01',
                   '1958-11-01', '1958-12-01',
                   ...
                   '2001-03-01', '2001-04-01', '2001-05-01', '2001-06-01',
                   '2001-07-01', '2001-08-01', '2001-09-01', '2001-10-01',
                   '2001-11-01', '2001-12-01'],
                  dtype='datetime64[ns]', name='date', length=526, freq='MS')



Looking at the index values, we can see that our time series now carries aggregated data on monthly terms, shown as `Freq: MS`. 

### Time-series Index Slicing for Data Selection

Slice our dataset to only retrieve data points that come after the year 1990.


```python
# Slice the timeseries to contain data after year 1990 
CO2[1990:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>co2</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1996-05-18</td>
      <td>365.7</td>
    </tr>
    <tr>
      <td>1996-05-25</td>
      <td>365.4</td>
    </tr>
    <tr>
      <td>1996-06-01</td>
      <td>364.8</td>
    </tr>
    <tr>
      <td>1996-06-08</td>
      <td>365.1</td>
    </tr>
    <tr>
      <td>1996-06-15</td>
      <td>365.2</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2001-12-01</td>
      <td>370.3</td>
    </tr>
    <tr>
      <td>2001-12-08</td>
      <td>370.8</td>
    </tr>
    <tr>
      <td>2001-12-15</td>
      <td>371.2</td>
    </tr>
    <tr>
      <td>2001-12-22</td>
      <td>371.3</td>
    </tr>
    <tr>
      <td>2001-12-29</td>
      <td>371.5</td>
    </tr>
  </tbody>
</table>
<p>294 rows × 1 columns</p>
</div>



Retrieve data starting from Jan 1990 to Jan 1991: 


```python
# Retrieve the data between 1st Jan 1990 to 1st Jan 1991
CO2['1990-01-01':'1991-01-01']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>co2</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1990-01-06</td>
      <td>353.4</td>
    </tr>
    <tr>
      <td>1990-01-13</td>
      <td>353.5</td>
    </tr>
    <tr>
      <td>1990-01-20</td>
      <td>353.8</td>
    </tr>
    <tr>
      <td>1990-01-27</td>
      <td>353.9</td>
    </tr>
    <tr>
      <td>1990-02-03</td>
      <td>354.1</td>
    </tr>
    <tr>
      <td>1990-02-10</td>
      <td>355.0</td>
    </tr>
    <tr>
      <td>1990-02-17</td>
      <td>354.8</td>
    </tr>
    <tr>
      <td>1990-02-24</td>
      <td>354.7</td>
    </tr>
    <tr>
      <td>1990-03-03</td>
      <td>355.7</td>
    </tr>
    <tr>
      <td>1990-03-10</td>
      <td>354.9</td>
    </tr>
    <tr>
      <td>1990-03-17</td>
      <td>355.8</td>
    </tr>
    <tr>
      <td>1990-03-24</td>
      <td>355.1</td>
    </tr>
    <tr>
      <td>1990-03-31</td>
      <td>355.9</td>
    </tr>
    <tr>
      <td>1990-04-07</td>
      <td>356.1</td>
    </tr>
    <tr>
      <td>1990-04-14</td>
      <td>355.9</td>
    </tr>
    <tr>
      <td>1990-04-21</td>
      <td>356.6</td>
    </tr>
    <tr>
      <td>1990-04-28</td>
      <td>356.1</td>
    </tr>
    <tr>
      <td>1990-05-05</td>
      <td>357.3</td>
    </tr>
    <tr>
      <td>1990-05-12</td>
      <td>357.0</td>
    </tr>
    <tr>
      <td>1990-05-19</td>
      <td>356.9</td>
    </tr>
    <tr>
      <td>1990-05-26</td>
      <td>357.1</td>
    </tr>
    <tr>
      <td>1990-06-02</td>
      <td>357.0</td>
    </tr>
    <tr>
      <td>1990-06-09</td>
      <td>356.6</td>
    </tr>
    <tr>
      <td>1990-06-16</td>
      <td>355.6</td>
    </tr>
    <tr>
      <td>1990-06-23</td>
      <td>355.5</td>
    </tr>
    <tr>
      <td>1990-06-30</td>
      <td>355.7</td>
    </tr>
    <tr>
      <td>1990-07-07</td>
      <td>355.5</td>
    </tr>
    <tr>
      <td>1990-07-14</td>
      <td>354.0</td>
    </tr>
    <tr>
      <td>1990-07-21</td>
      <td>354.5</td>
    </tr>
    <tr>
      <td>1990-07-28</td>
      <td>354.7</td>
    </tr>
    <tr>
      <td>1990-08-04</td>
      <td>353.5</td>
    </tr>
    <tr>
      <td>1990-08-11</td>
      <td>353.2</td>
    </tr>
    <tr>
      <td>1990-08-18</td>
      <td>352.9</td>
    </tr>
    <tr>
      <td>1990-08-25</td>
      <td>352.0</td>
    </tr>
    <tr>
      <td>1990-09-01</td>
      <td>350.9</td>
    </tr>
    <tr>
      <td>1990-09-08</td>
      <td>350.7</td>
    </tr>
    <tr>
      <td>1990-09-15</td>
      <td>351.3</td>
    </tr>
    <tr>
      <td>1990-09-22</td>
      <td>350.9</td>
    </tr>
    <tr>
      <td>1990-09-29</td>
      <td>350.9</td>
    </tr>
    <tr>
      <td>1990-10-06</td>
      <td>351.1</td>
    </tr>
    <tr>
      <td>1990-10-13</td>
      <td>351.0</td>
    </tr>
    <tr>
      <td>1990-10-20</td>
      <td>351.4</td>
    </tr>
    <tr>
      <td>1990-10-27</td>
      <td>351.4</td>
    </tr>
    <tr>
      <td>1990-11-03</td>
      <td>352.1</td>
    </tr>
    <tr>
      <td>1990-11-10</td>
      <td>352.6</td>
    </tr>
    <tr>
      <td>1990-11-17</td>
      <td>353.0</td>
    </tr>
    <tr>
      <td>1990-11-24</td>
      <td>353.1</td>
    </tr>
    <tr>
      <td>1990-12-01</td>
      <td>353.6</td>
    </tr>
    <tr>
      <td>1990-12-08</td>
      <td>354.0</td>
    </tr>
    <tr>
      <td>1990-12-15</td>
      <td>353.8</td>
    </tr>
    <tr>
      <td>1990-12-22</td>
      <td>354.5</td>
    </tr>
    <tr>
      <td>1990-12-29</td>
      <td>354.8</td>
    </tr>
  </tbody>
</table>
</div>



## Missing Values

Find the total number of missing values in the dataset.


```python
# Find the total number of missing values in the time series
CO2.isna().sum()
```




    co2    59
    dtype: int64



Remember that missing values can be filled in a multitude of ways. 

- Replace the missing values in `CO2_monthly_mean` with a previous valid value 
- Next, check if your attempt was successful by checking for number of missing values again 


```python
# Perform backward filling of missing values
CO2_final = CO2.ffill()

# Find the total number of missing values in the time series
CO2_final.isna().sum()
```




    co2    0
    dtype: int64



Great! Now your time series data are ready for visualization and further analysis.

## Summary

In this introductory lab, you learned how to create a time series object in Python using Pandas. You learned how to check timestamp values as the data index and you learned about basic data handling techniques for time-series data for further analysis.
