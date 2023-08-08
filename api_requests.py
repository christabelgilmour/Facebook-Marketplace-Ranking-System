import requests

response = requests.get("http://ec2-3-253-124-104.eu-west-1.compute.amazonaws.com:8080/healthcheck")
print(response.json())

url = "http://ec2-3-253-124-104.eu-west-1.compute.amazonaws.com:8080/predict/image"
files = {"image": open("./example_image.png", "rb")}

response = requests.post(url, files=files)
print(response.json())

url = "http://ec2-3-253-124-104.eu-west-1.compute.amazonaws.com:8080/predict/embedding"
files = {"image": open("./example_image.png", "rb")}

response = requests.post(url, files=files)
print(response.json())


