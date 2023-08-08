# Facebook Marketplace Recommendation Ranking System

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

## Milestone 1

First, I set up a github repository and an AWS cloud account.

## Milestone 2

I downloaded the datasets from AWS and clean the tabular dataset by removing all null values and ensuring the prices features are in numerical format. Once completed, I extract the labels for each classification category, by creating an encoder to map each category to a value. I then merge the tabular data and image data so I have a label for each image.

## Milestone 3 

Now the image dataset has to be cleaned to ensure all the images are the same size and have the same number of channels.

<img width="974" alt="Screenshot 2023-07-14 at 11 51 33" src="https://github.com/christabelgilmour/facebook-marketplaces-recommendation-ranking-system/assets/113252944/de7428c7-4fc1-437d-89a0-ab44c26d3e8e">


## Milestone 4 

Once both the tabular and image datasets are cleaned, I created a PyTorch dataset.

<img width="992" alt="Screenshot 2023-07-14 at 11 45 13" src="https://github.com/christabelgilmour/facebook-marketplaces-recommendation-ranking-system/assets/113252944/dea37a3f-b4d8-4e32-ac05-1c25124e78cb">



Next, I used transfer learning to finetune ResNet-50 to model a CNN.

<img width="990" alt="Screenshot 2023-07-14 at 11 53 14" src="https://github.com/christabelgilmour/facebook-marketplaces-recommendation-ranking-system/assets/113252944/5e580e61-12d5-42c9-8662-4c5a73820230">


To train the model, I randomly split the dataset into training, testing and validation sets. Then I train the model over 9 epochs.

<img width="985" alt="Screenshot 2023-07-14 at 11 54 52" src="https://github.com/christabelgilmour/facebook-marketplaces-recommendation-ranking-system/assets/113252944/725eb303-a73e-44f6-b88e-086225f16a82">


I visualise and evaluate the model's performance by plotting the epoch loss and epoch accuracy on Tensorboard.

<img width="974" alt="Screenshot 2023-07-14 at 11 56 30" src="https://github.com/christabelgilmour/facebook-marketplaces-recommendation-ranking-system/assets/113252944/86ebb23d-80ab-43b8-a019-38975901ba74">

<img width="979" alt="Screenshot 2023-07-14 at 11 57 10" src="https://github.com/christabelgilmour/facebook-marketplaces-recommendation-ranking-system/assets/113252944/2c6995b7-a114-4db5-a737-e28251de3026">


Once the model has been trained and the weights have been saved, I use the feature extraction model to get vector embeddings for each image in the training dataset. These are saved in a dictionary to be used in the Faiss search index.

<img width="976" alt="Screenshot 2023-07-14 at 11 59 11" src="https://github.com/christabelgilmour/facebook-marketplaces-recommendation-ranking-system/assets/113252944/8396ad85-4ab0-4364-b92f-32df90f451a3">


## Milestone 5

I built a Faiss Model by creating a matrix of each image embedding I saved.

<img width="976" alt="Screenshot 2023-07-14 at 12 00 09" src="https://github.com/christabelgilmour/facebook-marketplaces-recommendation-ranking-system/assets/113252944/8cba149b-a57c-4f55-b76e-b9bd429a3b05">


This can now be used to search for similar vectors for any given images set of features. 

<img width="987" alt="Screenshot 2023-07-14 at 12 00 33" src="https://github.com/christabelgilmour/facebook-marketplaces-recommendation-ranking-system/assets/113252944/832d2fd5-38eb-41f7-ac3b-4e214500fcf8">



## Milestone 6 

Using FastAPI, I created two API endpoints to retrieve the predicted category and 5 k-nearest neighbour vectors of an input image, by applying the necessary image transformations and feeding the image through my CNN and Faiss Model. 

<img width="982" alt="Screenshot 2023-07-14 at 12 01 21" src="https://github.com/christabelgilmour/facebook-marketplaces-recommendation-ranking-system/assets/113252944/573f15f7-c0fe-4a34-a971-1f4bcc9c32f6">


To deploy the API, I built a docker image within my EC2 instance.

<img width="889" alt="Screenshot 2023-07-14 at 11 27 49" src="https://github.com/christabelgilmour/facebook-marketplaces-recommendation-ranking-system/assets/113252944/8bcea105-440a-4a6d-8bdd-cf15bfd3ad89">


After running the docker image, I test the API by sending requests to the two endpoints.

<img width="911" alt="Screenshot 2023-07-14 at 11 25 22" src="https://github.com/christabelgilmour/facebook-marketplaces-recommendation-ranking-system/assets/113252944/5d3724a3-ebc8-4215-b4de-c617896b9d7a">


The results shown is the returned predicted category and k-nearest neighbours for two example images.

<img width="1109" alt="Screenshot 2023-07-14 at 11 38 53" src="https://github.com/christabelgilmour/facebook-marketplaces-recommendation-ranking-system/assets/113252944/4e534f3c-39c1-4dcf-b13a-c68a315fb6c9">



