import requests

# Define the URL and the data
url = 'http://127.0.0.1:5000/predict_score'
data = {
    "Number of subscribers": 566105142,
    "followers": 90091772,
    "average_likes": 21984,
    "average_comments": 7208,
    "average_shares": 3155,
    "budget_per_video": 109502,
    "engagement_rate": 3.8,
    "follower_growth_rate": 6.2,
    "primary_platform_Facebook": 1,
    "primary_platform_Instagram": 0,
    "primary_platform_YouTube": 0,
    "Targetaudience_gender_Female": 0,
    "Targetaudience_gender_Male": 0,
    "Targetaudience_gender_Others": 1,
    "product_category_Accessories": 1,
    "product_category_Clothing": 0,
    "product_category_Cosmetics": 0,
    "Geographical location: state of India_Gujarat": 0,
    "Geographical location: state of India_Kerala": 0,
    "Geographical location: state of India_Maharashtra": 0,
    "Geographical location: state of India_Punjab": 0,
    "Geographical location: state of India_Rajasthan": 0,
    "Geographical location: state of India_Sikkim": 1,
    "Geographical location: state of India_Tamil Nadu": 0,
    "LL_age": 25,
    "UL_age": 35,
    "campaign duration_transform": 9
    }

# Send the POST request
response = requests.post(url, json=data)

# Print the prediction result
print(response.json())
