import uvicorn
from fastapi import FastAPI

# ML Packages
import joblib,os

# Vectorizer
gender_vectorizer = open("Models/gender_vectorizer.pkl","rb")
gender_cv = joblib.load(gender_vectorizer)

# Models
gender_nv_model = open("Models/naivebayesgendermodel.pkl","rb")
gender_clf = joblib.load(gender_nv_model)


# Init App
app = FastAPI()

# Routes
@app.get('/')
async def index():
	return {"text":"Hello from Fast API "}


@app.get('/items/{name}')
async def get_items(name):
	return {"name":name}



# ML Aspects
@app.get('/predict/{name}')
async def predict(name):
	vectorized_name = gender_cv.transform([name]).toarray()
	prediction = gender_clf.predict(vectorized_name)
	if prediction[0] == 0:
		result = "female"
	else:
		result = "male"

	return {"original name":name, "prediction":result}





if __name__ == '__main__':
	uvicorn.run(app,host="127.0.0.1",port=8000)
