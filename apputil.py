import pandas as pd
import plotly.express as px
import numpy as np

def survival_demographics():
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    
    bins = [0, 12, 19, 59, 100]
    labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
    
    result = df.groupby(['Pclass', 'Sex', 'age_group'], observed=False, dropna=True).agg(
        n_passengers=('Survived', 'size'),
        n_survivors=('Survived', 'sum')
    ).reset_index()
    
    result['n_passengers'] = result['n_passengers'].astype(int)
    result['n_survivors'] = result['n_survivors'].fillna(0).astype(int)
    result['survival_rate'] = (result['n_survivors'] / result['n_passengers']).fillna(0.0)
    
    result = result.sort_values(['Pclass', 'Sex', 'age_group']).reset_index(drop=True)
    
    return result

def visualize_demographic():
    df = survival_demographics()
    df_clean = df[df['n_passengers'] > 0].copy()
    
    fig = px.bar(
        df_clean,
        x='age_group',
        y='survival_rate',
        color='Sex',
        facet_col='Pclass',
        barmode='group',
        title='Survival Rates by Class, Sex, and Age Group',
        labels={
            'survival_rate': 'Survival Rate',
            'age_group': 'Age Group',
            'Pclass': 'Passenger Class'
        },
        color_discrete_map={'male': '#1f77b4', 'female': '#ff7f0e'}
    )
    
    fig.update_yaxes(range=[0, 1])
    return fig

def family_groups():
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    
    result = df.groupby(['family_size', 'Pclass']).agg(
        n_passengers=('Fare', 'count'),
        avg_fare=('Fare', 'mean'),
        min_fare=('Fare', 'min'),
        max_fare=('Fare', 'max')
    ).reset_index()
    
    result = result.sort_values(['Pclass', 'family_size']).reset_index(drop=True)
    
    return result

def last_names():
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    
    df['LastName'] = df['Name'].str.split(',').str[0]
    
    return df['LastName'].value_counts()

def visualize_families():
    df = family_groups()
    
    fig = px.scatter(
        df,
        x='family_size',
        y='avg_fare',
        color='Pclass',
        size='n_passengers',
        title='Average Fare by Family Size and Passenger Class',
        labels={
            'family_size': 'Family Size',
            'avg_fare': 'Average Fare ($)',
            'Pclass': 'Class',
            'n_passengers': 'Number of Passengers'
        },
        color_continuous_scale='Viridis'
    )
    
    return fig

def determine_age_division():
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    
    class_medians = df.groupby('Pclass')['Age'].transform('median')
    
    df['older_passenger'] = None
    mask = df['Age'].notna()
    df.loc[mask, 'older_passenger'] = df.loc[mask, 'Age'] > class_medians[mask]
    
    return df

def visualize_age_division():
    df = determine_age_division()
    df_clean = df.dropna(subset=['Age', 'older_passenger']).copy()
    
    summary = df_clean.groupby(['Pclass', 'older_passenger', 'Survived']).size().reset_index(name='count')
    
    fig = px.bar(
        summary,
        x='Pclass',
        y='count',
        color='Survived',
        facet_col='older_passenger',
        barmode='group',
        title='Survival Count by Class and Age Division (Above/Below Class Median)',
        labels={
            'Pclass': 'Passenger Class',
            'count': 'Number of Passengers',
            'Survived': 'Survived',
            'older_passenger': 'Older than Class Median'
        },
        color_discrete_map={0: '#d62728', 1: '#2ca02c'}
    )
    
    return fig