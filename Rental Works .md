# Import pandas
import pandas as pd

# Import train test split
from sklearn.model_selection import train_test_split

# Import one hot encoder
from sklearn.preprocessing import OneHotEncoder

# Import column transformer
from sklearn.compose import ColumnTransformer

# Import pipeline
from sklearn.pipeline import Pipeline

# Import linear regression model
from sklearn.linear_model import LinearRegression

# Import evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Housing.csv")

# Define X and y
X = df.drop('price', axis=1)
y = df['price']

# Select categorical columns
cat_cols = X.select_dtypes(include='object').columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ],
    remainder='passthrough'
)

# Build pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
