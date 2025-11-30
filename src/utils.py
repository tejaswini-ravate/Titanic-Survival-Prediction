# src/utils.py
import pandas as pd
import re

def extract_title(name: str):
    """Extract title from a passenger Name field."""
    if pd.isna(name):
        return "Unknown"
    m = re.search(r',\s*(.*?)\.', name)
    if m:
        title = m.group(1)
        # normalization / grouping
        if title in ['Mlle', 'Ms']:
            return 'Miss'
        if title == 'Mme':
            return 'Mrs'
        if title in ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']:
            return 'Rare'
        return title
    return 'Unknown'

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple features used in the pipeline:
    - Title, FamilySize, IsAlone, Deck
    - Fillna for Fare/Age/Embarked and convert Sex -> numeric
    Returns a new DataFrame copy.
    """
    df = df.copy()
    # Title from Name
    if 'Name' in df.columns:
        df['Title'] = df['Name'].apply(extract_title)
    else:
        df['Title'] = 'Unknown'

    # Family features
    df['FamilySize'] = df.get('SibSp', 0) + df.get('Parch', 0) + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Deck from Cabin (first letter) or 'U' if missing
    df['Deck'] = df.get('Cabin').fillna('U').astype(str).apply(lambda x: x[0] if x != 'U' else 'U')

    # Fill missing numerical values
    """if 'Fare' in df.columns:
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
    if 'Age' in df.columns:
        df['Age'].fillna(df['Age'].median(), inplace=True)
    if 'Embarked' in df.columns:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)"""

    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Encode Sex numeric for simple fallback usage if needed
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    return df
