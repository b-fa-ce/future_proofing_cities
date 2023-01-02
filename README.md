# Future proofing cities
a 10 day project with the Le Wagon Data Science & Machine Learning bootcamp


<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="fpc_logo.png" alt="Logo" width="175" height="175">
  </a>
</div>


## Description
This repository is readily available for anyone wanting to evaluate heat distribution in an urban landscape. The aim of the project is to use a convolutional neural network to predict heat islands within
a city, which we defined as areas of relatively high land surface temperature (LST).
This Neural Network takes 15 features determining a city's topography, land cover types, building height and density and will predict the difference to the mean temperature for each pixel.
Pixels are defined as 70*70m and are fed into the Network as a tensorflow object of shape (1, 15)

Features:
 - Average Building Density/pixel
 - Average Building Height/pixel
 - Elevation
 - Landuse Type (12 categories)

Target:
 - Difference to the Mean of Land Surface Temperature/ pixel

## Installation & usage

**Data**
 - Preprocessed Data for Land Cover for Paris and Berlin:
 data/processed_data/Berlin/Berlin_landuse.csv

 - Preprocessed Data for Elevation:
 data/processed_data/Berlin/Berlin.csv

 - Preprocessed Data for Building Height and Density
 ohsome API on building data

**Model**
modules/ml_logic/model.py

### Training the model

- Install dependencies
```
make install
```

- Train on Paris data
```
make run_train
```


### Predicting using docker

- Copy content from `.env-sample` to `.env` and update your relevant information
- Run
```
pyenv allow
```
- Build docker image
```
docker build -t $IMAGE:prod .
```
- Run docker image on `port 8000`
```
docker run -it -e PORT=8000 -p 8000:8000 --env-file .env $IMAGE:prod
```
- Access the `docs` on `localhost:8000/docs`

### Streamlit frontend

Our Git Repository which contains the information for the deployment of our user interface:

https://github.com/b-fa-ce/future-proofing-cities-frontend


## Project Contributors:
 - Bruno Faigle-Cedzich
 - Matt Hall
 - Afanasis Kiurdzhyiev
 - Leah Rothschild

Project created and developed in the context of finalising our Data Science & Machine Learning Course with Le Wagon
