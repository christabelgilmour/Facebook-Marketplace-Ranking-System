import api_requests
url = "https://projectstack-9f1749cc-4ba8-4037-ade0-414-datalake-xz4xwdslsft7.s3.eu-west-1.amazonaws.com/api_template.py"
response = api_requests.get(url)
with open('api_template.py', 'wb') as f:
    f.write(response.content)