# Importing all necessary libraries
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier

# Defining the URL for Premier League standings
standing_url = "https://fbref.com/en/comps/9/Premier-League-Stats"

# Creating an empty list to store the match data
all_matches = []

# Defining a list of years to scrape data for
years = list(range(2022, 2020, -1))

# Fetching the Premier League data from the URL
pl_data = requests.get(standing_url)
pl_html = pl_data.text
soup = BeautifulSoup(pl_html)

# Selecting the Premier League standing table
pl_standing_table = soup.select("table.stats_table")[0]

# Extracting the links to individual team stats pages
pl_links = pl_standing_table.find_all("a")
pl_links = [l.get("href") for l in pl_links]
pl_links = [l for l in pl_links if '/squads/' in l]
team_urls = [f"https://fbref.com{l}" for l in pl_links]

# Iterating through each team's stats page
for team_url in team_urls:
    # Extracting the team name from the URL
    team_name = team_url.split("/")[-1].replace("Stats", "").replace("-", " ")

    # Fetching data from the specific team's page
    pl_team_data = requests.get(team_url)
    pl_team_link = pl_team_data.text
    time.sleep(30)

    # Reading the matches' data from the team's page
    matches = pd.read_html(pl_team_link, match="Scores & Fixtures")[0]

    # Creating a BeautifulSoup object for the team's page
    soup2 = BeautifulSoup(pl_team_link)

    # Extracting the links related to shooting statistics
    links2 = soup2.find_all("a")
    links2 = [l.get("href") for l in links2]
    links2 = [l for l in links2 if l and "all_comps/shooting/" in l]

    # Iterating through the shooting statistics links
    for link_ in links2:
        # Fetching data from the shooting statistics page
        data3 = requests.get(f"https://fbref.com{link_}")
        time.sleep(30)
        shooting = pd.read_html(data3.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()

        # Merging match data with shooting statistics data based on the date
        team_data = None
        if 'Date' in shooting.columns:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        else:
            continue

        # Filtering the data for Premier League matches
        team_data = team_data[team_data["Comp"] == "Premier League"]

        # Adding season and team name columns
        team_data["Season"] = years
        team_data["Team"] = team_name

        # Appending the team's match data to the list
        all_matches.append(team_data)

# Concatenating all match data into a single DataFrame
match_df = pd.concat(all_matches)

# Converting column names to lowercase
match_df.columns = [c.lower() for c in match_df.columns]

# Saving the DataFrame to a CSV file
match_df.to_csv("matches.csv")

# Loading the match data from a CSV file
matches = pd.read_csv("/Users/swaminathang/Downloads/matches.csv", index_col=0)

# Counting the number of occurrences of each team in the dataset
team_counts = matches["team"].value_counts()

# Filtering and viewing data for matches involving "Liverpool"
liverpool_matches = matches[matches["team"] == "Liverpool"]

# Counting the occurrences of each "round" in the dataset
round_counts = matches["round"].value_counts()

# Converting the "date" column to a datetime format
matches["date"] = pd.to_datetime(matches["date"])

# Encoding categorical features: "venue_code" and "opp_code"
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes

# Assigning target since it's assumed to be the outcome variable
target = "target"

# Extracting the hour from the "time" column and convert it to an integer
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")

# Extracting the day of the week and encode it as "day_code"
matches["day_code"] = matches["date"].dt.dayofweek

# Mapping result labels ("L", "D", "W") to numeric values (0, 1, 2)
result_mapping = {"L": 0, "D": 1, "W": 2}
matches["target"] = matches["result"].map(result_mapping)

# Creating a RandomForestClassifier with specified parameters
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

# Splitting the data into training and test sets based on a date threshold
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']

# Defining predictors for the model
predictors = ["venue_code", "opp_code", "day_code"]

# Fitting the RandomForestClassifier on the training data
rf.fit(train[predictors], train["target"])

# Making predictions on the test data
preds = rf.predict(test[predictors])

# Calculating and printing the accuracy of the model
acc = accuracy_score(test["target"], preds)
print("Accuracy:", acc)

# Defining the hyperparameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'min_samples_split': [2, 5, 10],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False],
}

# Creating a GridSearchCV object with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')

# Fitting the grid search to the data
grid_search.fit(train[predictors], train[target])

# Getting the best hyperparameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:")
print(best_params)

# Getting the best model from the grid search
best_rf_model = grid_search.best_estimator_

# Evaluating the best model on the test data
test_preds = best_rf_model.predict(test[predictors])
test_acc = accuracy_score(test[target], test_preds)
print("Test Accuracy with Best Model:", test_acc)

# Creating a DataFrame to combine actual and predicted values
combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))

# Generating a cross-tabulation of actual vs. predicted values
confusion_matrix = pd.crosstab(index=combined["actual"], columns=combined["prediction"])

# Calculating precision using a weighted average
test_precision = precision_score(test["target"], preds, average = "weighted")

# Calculating recall
test_recall = recall_score(test["target"], test_preds, average="weighted")
print("Test Recall:", test_recall)

# Calculating F1 score
test_f1 = f1_score(test["target"], test_preds, average="weighted")
print("Test F1 Score:", test_f1)

# Creating a dataframe to display the results
initial_results_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [acc, test_precision, test_recall, test_f1]
})

# Displaying the results
print(initial_results_df)

# Grouping the matches data by "team"
grouped_matches = matches.groupby("team")

# Fetching the group of matches for "Manchester City"
group = grouped_matches.get_group("Manchester City")

# Defining a function to calculate rolling averages for specific columns
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed="left").mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

# Defining columns and new column names for rolling averages
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

# Calculating rolling averages for the "Manchester City" group
rolling_averages(group, cols, new_cols)

# Grouping the matches data by "team" and applying rolling averages function
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))

# Dropping the "team" level from the index
matches_rolling = matches_rolling.droplevel("team")

# Resetting the index of the DataFrame
matches_rolling.index = range(matches_rolling.shape[0])

# Defining a function to make predictions and calculate the precision
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
    precision = precision_score(test["target"], preds, average="weighted")
    return combined, precision

# Making predictions and calculating precision for the rolling averages data
combined, precision = make_predictions(matches_rolling, predictors + new_cols)

# Merging additional columns into the combined DataFrame
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

# Defining a custom dictionary class for mapping values
class MissingDict(dict):
    __missing__ = lambda self, key: key

# Defining a dictionary for mapping team names
map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)

# Mapping team names using the custom dictionary for better understanding
combined["new_team"] = combined["team"].map(mapping)

# Merging the combined DataFrame with itself on date and "new_team" columns
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])

# Filtering and counting matches with specific predictions and printing them
filtered_matches = merged[(merged["prediction_x"] == 2) & (merged["prediction_y"] == 0)]
value_counts = filtered_matches["actual_x"].value_counts()
print(value_counts)

# Calculating the ratio of specific predictions
ratio = len(filtered_matches) / len(merged)
print(ratio)

# Splitting your data into training and test sets
X = matches_rolling[predictors].values
y = matches_rolling[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the Recurrent Neural Network (RNN) model
rnn_model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),         # The input layer
    layers.Reshape((1, X_train.shape[1])),           # Reshaping the input for RNN
    layers.LSTM(64, activation='relu', return_sequences=True), # The LSTM layer with return_sequences=True
    layers.LSTM(32, activation='relu'),              # The LSTM layer without return_sequences
    layers.Dense(3, activation='softmax')            # The output layer with 3 units (for win, draw, loss) and softmax activation
])

# Compiling the RNN model
rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the RNN model with a larger number of epochs and monitoring the validation loss
rnn_history = rnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluating the RNN model on the test set
test_loss, test_accuracy = rnn_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Making predictions on new data
rnn_predictions = rnn_model.predict(X_test)
print(f"RNN Predictions:", rnn_predictions)

# Building the Feedforward Neural Network (FNN) model
fnn_model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # The input layer
    layers.Dense(128, activation='relu'),     # The hidden layer with 128 units and ReLU activation
    layers.Dropout(0.5),                      # The dropout layer to reduce overfitting
    layers.Dense(64, activation='relu'),      # Another hidden layer with 64 units and ReLU activation
    layers.Dense(3, activation='softmax')     # The output layer with 3 units (for win, draw, loss) and softmax activation
])

# Compiling the FNN model
fnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the FNN model with a larger number of epochs and monitoring the validation loss
fnn_history = fnn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Evaluating the FNN model on the test set
test_loss, test_accuracy = fnn_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Making predictions on new data
fnn_predictions = fnn_model.predict(X_test)
print(f"FNN Predictions:", fnn_predictions)