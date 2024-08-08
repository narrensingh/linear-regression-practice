import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('housing.csv')
df = df[['population','households']]

def normalization(array):
    mean = array.mean()
    array_trans = array - mean
    sd = array_trans.std()
    return array_trans/sd
