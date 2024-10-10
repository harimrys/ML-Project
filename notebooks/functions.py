import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

def dropcolumns(df, column1, column2, column3):
    """
    Drop the specified columns from a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to drop columns from.
    column1 : str
        The first column to drop.
    column2 : str
        The second column to drop.
    column3 : str
        The third column to drop.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with the specified columns dropped.
    """
    # Drop the specified columns from the DataFrame
    df = df.drop(columns = [column1, column2, column3], inplace = True)
    # Reset the index
    df.reset_index(drop = True, inplace = True)
    return df

def mapeo_gender(df, columns):
    gender_map = {"Female": True, "Male": False}
    df[columns] = df[columns].map(gender_map)
    return df

def mapeo(df, columns):
    map_total = {"Yes": True, "No": False}
    df[columns] = df[columns].map(map_total)
    return df


def bar_plot(df, columns):
    plt.figure(figsize=(8, 5))
    sns.countplot(data = df, x= columns, palette='Blues')
    plt.title('Frecuencia de Categorías')
    plt.xlabel('Categoría')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=0)
    plt.show()

    
def correlacion(features):

    # Calculate the correlation matrix
    correlation_matrix = np.abs(features.corr())
    # Create the heatmap using Plotly Express
    fig = px.imshow(correlation_matrix,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    color_continuous_scale='RdBu_r',  # Red-Blue diverging color scale
                    zmin=-1,
                    zmax=1,
                    aspect="auto",
                    title='')
    # Update the layout for better readability
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        xaxis={'side': 'top'},  # Move x-axis labels to the top
        width=800,
        height=700
    )
    # Add correlation values as text annotations
    for i, row in enumerate(correlation_matrix.values):
        for j, value in enumerate(row):
            fig.add_annotation(
                x=correlation_matrix.columns[j],
                y=correlation_matrix.columns[i],
                text=f"{value:.2f}",
                showarrow=False,
                font=dict(size=8)
            )
    # Show the plot
    fig.show()


def graficos(df, columns):
    df[columns].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#CDB4DB', '#FFC8DD', '#BDE0FE'], figsize=(6,6))
    plt.ylabel('')
    plt.title(columns)
    plt.show()