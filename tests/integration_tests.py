"""This module contains class `TestMissForest`."""

__author__ = "Yuen Shing Yan Hindy"

import unittest
import pandas as pd
import numpy as np
from scipy.stats import norm, binom
from sklearn.model_selection import train_test_split
from src.missforest.missforest import MissForest


class IntegrationTests(unittest.TestCase):
    """Tests MissForest as a combined entity."""
    def setUp(self):
        """Special method that is automatically called before each test
        method is executed. It is used to set up `MissForest` instance that
        is shared across multiple tests."""

        while True:
            # make synthetic datasets
            # seed to follow along
            np.random.seed(1234)

            # generate 1000 data points
            n = np.arange(1000)

            # create correlated, random variables
            a = 2
            b = 1 / 2
            eps = np.array([norm(0, np.random.choice(np.arange(50))).rvs() for _ in n])
            y = (a + b * n + eps) / 100
            x = (n + norm(10, np.random.choice(np.arange(250))).rvs(len(n))) / 100

            # add missing values
            y[binom(1, 0.4).rvs(len(n)) == 1] = np.nan

            # convert to dataframe
            df = pd.DataFrame({"y": y, "x": x})

            self.train, self.test = train_test_split(df, test_size=.3)

            if (
                    self.train.isnull().sum().sum() > 0 and
                    self.test.isnull().sum().sum() > 0
            ):
                break

    def test_integration1(self):
        """Tests if MissForest can run properly when its `fit` and 
        `transform` methods are called."""
        mf = MissForest()
        mf.fit(self.train)
        train_imputed = mf.transform(self.test)
        test_imputed = mf.transform(self.test)

        self.assertEqual(train_imputed.isnull().sum().sum(), 0)
        self.assertEqual(test_imputed.isnull().sum().sum(), 0)

    def test_integration2(self):
        """Tests if MissForest can run properly when its `fit` and 
        `transform` methods are called."""
        mf = MissForest()
        train_imputed = mf.fit_transform(self.train)
        test_imputed = mf.transform(self.test)

        self.assertEqual(train_imputed.isnull().sum().sum(), 0)
        self.assertEqual(test_imputed.isnull().sum().sum(), 0)

    def test_integration3(self):
        """Tests if MissForest can run properly when its `fit` and 
        `transform` methods are called."""
        mf = MissForest()
        mf.fit(self.train)
        train_imputed = mf.fit_transform(self.train)
        test_imputed = mf.fit_transform(self.test)

        self.assertEqual(train_imputed.isnull().sum().sum(), 0)
        self.assertEqual(test_imputed.isnull().sum().sum(), 0)

    def test_integration4(self):
        """Tests if MissForest can run properly when its `fit` and 
        `transform` methods are called."""
        mf = MissForest()
        mf.fit(self.train)
        test_imputed = mf.fit_transform(self.test)
        train_imputed = mf.fit_transform(self.train)

        self.assertEqual(train_imputed.isnull().sum().sum(), 0)
        self.assertEqual(test_imputed.isnull().sum().sum(), 0)

    @staticmethod
    def test_integration_unseen_categories():
        """Tests if MissForest can run properly when are unseen categories."""
        df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [0, 1, 2, 0]})
        train = df[df["B"] != 0].copy()
        test = df[df["B"] == 0].copy()
        train.iloc[0, 0] = np.nan
        test.iloc[0, 0] = np.nan

        mf = MissForest(categorical=["B"])
        mf.fit(train)
        mf.transform(train)
        mf.transform(test)
