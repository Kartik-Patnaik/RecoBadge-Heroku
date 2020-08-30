# RecoBadge-Heroku
RecoBadge – An AI based badge recommendation engine based on historical badge pattern and their business relevance  

1.Increase penetration of badge programme across ranks and SLs as current trend shows a highly skewed distribution
2.Increase interest level among employees to complete badges by recommending right set of badges based on business relevance and their learning experience

procfile: is the app generation file for HEROKU
requirements.txt is the modules required to run it in server
app.py: is the application file to run in server
badge1.py: The the Neural Network file
encoder.pkl,nnclassifier.pkl,root_data.pkl,saved_data.pkl and scaler.pkl are the pickle file generated from the neural network and few are saved as cache by us to add intelligence.

