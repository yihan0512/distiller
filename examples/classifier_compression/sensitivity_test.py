from numpy.core.arrayprint import set_string_function
from distiller.sensitivity import *
import pandas as pd


if __name__ == '__main__':
    fname = 'logs/2021.05.30-022016/sensitivity.csv'
    sensitivities = csv_to_sensitivities(fname)
    sensitivities_to_png(sensitivities, 'test.png')
    sensitivities_analysis(sensitivities, 80)