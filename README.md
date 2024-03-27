1. Retrieve access token (authentication):
   curl --location 'localhost:5000/login' \
   --header 'Content-Type: application/json' \
   --data '{
   "username": "admin",
   "password": "admin"
   }'

2. Use Random Forest Algorithm
   curl --location 'localhost:5000/predict' \
   --header 'Content-Type: application/json' \
   --header 'Authorization: Bearer YOUR_JWT' \
   --data '{
   "algorithm": "RandomForest",
   "randomState": 42
   }'

3. Use Decision Tree Algorithm
   curl --location 'localhost:5000/predict' \
   --header 'Content-Type: application/json' \
   --header 'Authorization: Bearer YOUR_JWT' \
   --data '{
   "algorithm": "DecisionTree",
   "randomState": 42
   }'