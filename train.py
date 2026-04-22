from sklearn.datasets import load_iris 
from sklearn.ensemble import RandomForestClassifier 
data = load_iris() 
X, y = data.data, data.target 
model = RandomForestClassifier() 
model.fit(X, y) 
print("Model trained successfully!") 