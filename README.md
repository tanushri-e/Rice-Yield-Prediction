# Deep Learning Based Yield Forecasting For Rice Varieties

## Overview
This project focuses on predicting rice yield in Tamil Nadu, specifically in the Thanjavur district, using a Deep Learning based approach. Long Short-Term Memory (LSTM) networks are applied to analyze sequential patterns in agricultural datasets which include NDVI and climate parameters. The model provides pre-harvest yield forecasts aimed at supporting farmers and agricultural stakeholders through data-driven decisions.

## Objectives
- Predict rice yield using historical NDVI and climate data.
- Provide an AI-powered decision-support system for sustainable agriculture.
- Improve food productivity and farmer livelihood through accurate forecasting.

## Key Features
- Time-series modeling using LSTM neural networks.
- Integration of NDVI satellite vegetation indices.
- Climate variables including rainfall and temperature.
- Visualization of predictions and model performance.
- Scalable and adaptable solution for other crops and regions.

## Technologies Used
- Python 3.10+
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn

## Dataset Description
|       Dataset     |                  Description                 |           Purpose               |
|-------------------|----------------------------------------------|---------------------------------|
| NDVI Data         | Satellite-derived vegetation strength values | Crop growth monitoring          |
| Climate Data      | Rainfall, Temperature, and Weather Records   | Environmental impact assessment |
| Rice Variety Data | District-level crop information              | Yield behavior modeling         |

## System Architecture
1. Data Collection
2. Data Preprocessing and Normalization
3. LSTM Model Training and Validation
4. Yield Prediction and Visualization

# Rice-Yield-Prediction Project

## Project Structure

Rice-Yield-Prediction/
├─ SourceCode/
│  ├─ data/
│  │  ├─ sample_ndvi_wide.csv
│  │  ├─ synthetic_rice_yield_data.csv
│  │  ├─ temp_climate.csv
│  │  └─ training_ready.csv
│  ├─ models/
│  │  ├─ labelencoder.pkl
│  │  ├─ lstm_rice_yield_model.h5
│  │  └─ scaler.pkl
│  ├─ app.py
│  ├─ train.py
│  ├─ preprocess.py
│  ├─ fine_tune_synthetic.py
│  ├─ utils.py
│  └─ requirements.txt
├─ Executable/
│  └─ No executable version available for this project.
├─ Documentation/
│  ├─ Project_Report.docx
│  ├─ Final_Viva_PPT.pptx
│  ├─ Journal_Paper.docx
│  └─ One_Page_Abstract.docx
├─ Video/
│  └─ Working_Demo.mp4
└─ README.md

## How to Run the Project

### Step 1: Install Required Packages

pip install -r requirements.txt


### Step 2: Run Model or Application

streamlit run app.py


### Output Information

* Rice yield prediction results in tons per hectare.
* Graphs displaying NDVI and climate trends.

## Applications

* Agricultural production planning
* Government procurement forecasting
* Farmer advisory and risk mitigation
* Early detection of yield decline

## Limitations and Future Scope

* Model accuracy depends on quality and resolution of data.
* Integration with live satellite data can be enhanced.
* Can be extended to district-wide or state-level scalable deployment.

## Authors

* Students Name: Tanushri E & Thiruselvi Thirunavukkarasu
* Register Numbers : 211423104684 & 211423104698
* Department of Computer Science and Engineering

## Guide & Project Co-ordinators

* Faculty Guide Name: Dr. Sathiya V  
* Project Co-ordinators: Dr. Subedha V & Mrs. S. Sharmila

## Institution

* Panimalar Engineering College

## Disclaimer
This project is developed solely for academic purposes as part of a college coursework. Redistribution or commercial usage of the project is not permitted.

## Acknowledgment

Special thanks to government agricultural datasets and open-source tools that aided this project.


