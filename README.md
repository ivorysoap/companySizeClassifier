## Background

*Under construction.*

`modelTrainer.py` builds a machine-learning model using a logistic regression classifier, if given a dataset with the filename `companies_sorted.csv`.

In our dataset of about 7,000,000 companies, the following table shows the approximate number of companies per each size range:

| size_range   | Number of companies |
|--------------|---------|
| 1 - 10       | 1674694 |
| 11 - 50      | 570370  |
| 51 - 200     | 181524  |
| 201 - 500    | 43749   |
| 501 - 1000   | 15647   |
| 1001 - 5000  | 13632   |
| 5001 - 10000 | 1855    |
| 10001+       | 1366    |

The dataset contains around 7 million companies, with attributes like company names, company locations, year of inception, size category, etc.000

`modelTrainer.py` creates a machine-learning model based on this data.  The model, given a **company name** and **year of inception**, tries its best to predict that company's **size category** (see above).

`companySizeEstimator.py` is the frontend for this model.

## Usage

`python modelTrainer.py [--testonly] [filename]`

The ``[--testonly]`` flag specifies to test an existing machine-learning model, instead of creating and training one. If using the `--testonly` flag, you have to specify a `[filename]` from which to load the existing model.

## Attribution

I wrote this from scratch, with the exception of some borrowed snippets from the Intro to Artifical Intelligence course at uOttawa (CSI 4506, Prof. Caroline Barrière) - specifically the train_model() method came from a homework assignment notebook.
