import os, sys, gc, warnings
import numpy as np
import pandas as pd
import time
import xgboost
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import statsmodels.api as sm

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.display import display, HTML
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import SVG

from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials, STATUS_FAIL, space_eval
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from functools import partial

pd.set_option('display.float_format', lambda x: '%.3f' % x)

plotly.offline.init_notebook_mode()
warnings.filterwarnings("ignore")
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('max_rows', None)
pd.set_option('max_columns', None)
pd.set_option('display.max_colwidth', 100)
display(HTML(data="""<style> div#notebook-container{width:95%;}</style>"""))
warnings.filterwarnings("ignore", category=DeprecationWarning) 