This document describes how the project code can be set up and run.

Step 1: Aquire data
    Our input dataset is included in /data/reuters_headlines.csv
    If you want to acquire the data yourself, it's available for download at:
        https://www.kaggle.com/datasets/notlucasp/financial-news-headlines/data
        Make sure you download the reuters_headlines.csv file, not any of the others, if you want to repeat our results.

Step 2: Run a Sentence Transformer Model on Data
    We applied the sentence-transformers/all-MiniLM-L6-v2 model to our data
        That model can be found here: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    To execute this code, run the process_text.py file found in the root directory
        Make sure to resolve any dependency issues first.
    On line 9 of process_text.py, make sure CSV file name reflects the location and name of the input dataset.
    Once completed, that file will output a file titled "normalized_output_file.csv"

    Note: this is a very computationally expensive step. You will want a good computer, preferably with CUDA installed, to run this part of the program.

    If you want to skip Steps 1 and 2, you can simply use the "normalized_output_file.csv" file provided in the /data directory

Step 3: run through Streamlit App
    To run the rest of our code, it's required to use Streamlit.

There are two ways to execute the remainder of this project:
1. Access our deployed Streamlit App
2. Deploy your own Streamlit App using our code

The easiest by far is Option 1, but Option 2 offers more control over the process.
    Additionally, a deployed Streamlit App sometimes crashes, and an admin is required to restart it. 
    If that occurs, option 2 will ensure a reliable way to execute the project



---------------------- Option 1: Access Deployed Streamlit App ----------------------

1. This project is deployed through Streamlit, so the primary mechanism to run the code is through a browser:
    URL: https://version2py-rcb6mw3nvfsltcqut8kguj.streamlit.app/

2. If you encounter this message "This app has gone to sleep due to inactivity. Would you like to wake it back up?":
    Select "Yes, get this app back up!"
    Wait for app to finish Startup process

3. To get started, the app requires a CSV data file to begin. 
    This file is our sentence data, processed through Sentence-BERT
    The file to upload is located in the /data directory, called normalized_output_file.csv

    Select "Browse files" on the left-hand sidebar under "Upload your CSV file"
        Navigate to "normalized_output_file.csv", select it, and wait for the file to upload.

4. Select hyperparameters
    After the file completes uploading, the visualization code will begin.
    However, at any point you can adjust the hyperparameters to whatever values you desire.

    Recommended hyperparameters:
        Number of samples: 32770 (MAX)
        Number of PCA Components: 50
        t-SNE Perplexity: 50
        Number of Clusters for Coloring: 50

5. Analyze results, including hover-text graph
    After selecting hyperparameters, wait for the model to complete execution.
        The code being executed at this point does the following:
            - Additional preprocessing of data
            - Applies PCA
            - Applies t-SNE
            - Applies KMeans
            - Creates a plot using PyPlot
            - Creates a download link for clustering data
            - Creates a plot using Plotly
    Once model execution is completed, these are the results:
        Graph 1: a PyPlot visualization of the model output, color-coded by assigned cluster
        Download link: an opportunity to view the tsne dimensions, assigned cluster, and headline for each data point
        Graph 2: a Plotly visualization of the model output, color-coded by assigned cluster.
            This graph allows you to mouse over the graph and read information about the data point. That information is:
                t-SNE dimensions, assigned cluster, and headline

This concludes Option 1 of executing this project: Access Deployed Streamlit App

---------------------- Option 2: Deploy New Streamlit App ----------------------

To create a new Streamlit App:
    - Sign in (create an account if necessary)
    - Click "Create app" in the top right
    - Select "Deploy a public app from a template"
    - Select Blank app

Modifications required to Streamlit App files:
    - Copy requirements.txt from /streamlit directory over to new deployment
    - Copy version2.py from /streamlit directory over to new deployment
    - Install / resolve any dependency issues
        When editing through CodeSpaces, access the terminal window and run the following:
            pip install matplotlib
            pip install scikit-learn
            pip install plotly

Run program

---------------------- Conclusion ----------------------

This concludes the description of how the project code can be set up and run.

If you have any questions, please reach out to us, we'll be happy to help troubleshoot!


