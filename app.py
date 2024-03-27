import pandas as pd
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Modeling: Decision Tree and Random Forest Algorithms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

app.config['JWT_SECRET_KEY'] = 'sxtnqJ5t^r^JdMVJOlV4z(RMfVxP1aSMt1'
jwt = JWTManager(app)

USER_CREDENTIALS = {
    'username': 'admin',
    'password': 'admin'
}


# Inloggningsruta
@app.route('/login', methods=['POST'])
def login():
    """
    Autentiseringsruta för att generera åtkomsttoken.
    """
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    username = request.json.get('username', None)
    password = request.json.get('password', None)

    if not username or not password:
        return jsonify({"msg": "Missing username or password"}), 401

    if username != USER_CREDENTIALS['username'] or password != USER_CREDENTIALS['password']:
        return jsonify({"msg": "Invalid username or password"}), 401

    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token), 200


@app.route('/predict', methods=['POST'])
@jwt_required()
def predict_survival_probability():
    # Ensure request has JSON data
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    algorithm = request.json.get('algorithm', None)
    randomState = request.json.get('randomState', None)

    if algorithm == "RandomForest":
        return jsonify(value=useRandomForestClassifier(randomState=randomState)), 200
    elif algorithm == "DecisionTree":
        return jsonify(value=useDecisionTreeClassifier(randomState=randomState)), 200


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 35

        elif Pclass == 2:
            return 30

        else:
            return 25

    else:
        return Age


def prepare_dataframe():
    # load and read the data
    df1 = pd.read_csv(r'C:\Users\Gaukhar\Python\mu23\Titanic-Dataset.csv')

    # Count the number of missing values in each column of the DataFrame df
    df1.isnull().sum()

    # Drop the 'Cabin' column from the DataFrame df
    df1.drop(columns=['Cabin'], axis=1, inplace=True)
    df1['Age'] = df1[['Age', 'Pclass']].apply(impute_age, axis=1)

    df1['Fare'] = df1['Fare'].fillna(df1['Fare'].mean())

    df1.isnull().sum()

    df1.drop(columns=['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

    df_new = pd.get_dummies(df1, columns=['Sex', 'Embarked'])

    df_new['Sex_female'] = df_new['Sex_female'].astype(int)
    df_new['Sex_male'] = df_new['Sex_male'].astype(int)
    df_new['Embarked_C'] = df_new['Embarked_C'].astype(int)
    df_new['Embarked_Q'] = df_new['Embarked_Q'].astype(int)
    df_new['Embarked_S'] = df_new['Embarked_S'].astype(int)
    return df_new


def useRandomForestClassifier(randomState):
    df_new = prepare_dataframe()

    # Splitting the dataset into features and target variable
    X = df_new.drop('Survived', axis=1)
    y = df_new['Survived']
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomState)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest Algorithm
    random_forest = RandomForestClassifier(randomState)

    # Training the Random Forest model
    random_forest.fit(X_train_scaled, y_train)
    # Making predictions RF
    y_pred_rf = random_forest.predict(X_test_scaled)
    # Evaluating RF
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print("Random Forest Accuracy:", rf_accuracy)
    return rf_accuracy


def useDecisionTreeClassifier(randomState):
    df_new = prepare_dataframe()
    # Splitting the dataset into features and target variable
    X = df_new.drop('Survived', axis=1)
    y = df_new['Survived']
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomState)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Decision Tree Algorithm

    decision_tree = DecisionTreeClassifier(random_state=randomState)
    # Training the Decision Tree model
    decision_tree.fit(X_train_scaled, y_train)

    # Making predictions DT
    y_pred_dt = decision_tree.predict(X_test_scaled)

    # Evaluating DT
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    print("Decision Tree Accuracy:", dt_accuracy)
    return dt_accuracy

