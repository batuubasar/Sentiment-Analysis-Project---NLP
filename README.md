# Sentiment Analysis Project / NLP 
This project focuses on a sentiment analysis problem within the field of Natural Language Processing (NLP), specifically analyzing product reviews written in Turkish on e-commerce platforms like Amazon TR.
The goal is to determine whether customer reviews are positive or negative, helping companies better understand customer feedback and improve their product development processes.

## Project Objective
In the e-commerce world, analyzing customer reviews is crucial. This project analyzes product reviews from e-commerce platforms in Turkey, allowing companies to evaluate the public perception of their products or services.

## Dataset and Processing
The dataset is composed of customer reviews gathered from various Turkish e-commerce platforms. These reviews underwent preprocessing steps such as removing punctuation, eliminating stopwords, and identifying word roots.

## Models Used
- **Machine Learning (Naive Bayes):** A probabilistic model based on Bayes' theorem, commonly used for text classification. We used CountVectorizer to convert text data into numerical vectors.
- **Deep Learning (LSTM):** We implemented an LSTM (Long Short-Term Memory) model using the Keras library to process sequential data (text data). LSTMs are particularly useful for capturing long-term dependencies in text.
- **Hyperparameter Optimization:** We used Keras Tuner to optimize the hyperparameters of the models, ensuring better performance.

## Project Workflow
1. **Data Preprocessing:** Cleaned the text data, converted to lowercase, removed stopwords, and tokenized.
2. **Model Training:** Trained both Naive Bayes and LSTM models.
3. **Model Comparison:** Compared the accuracy of both models to determine which performs better.
4. **Results:** Evaluated the overall performance of the models and presented the results graphically.

## Libraries and Tools Used
- **Keras:** For the LSTM model.
- **Scikit-learn:** For Naive Bayes and data preprocessing tools.
- **Zeyrek MorphAnalyzer:** To analyze word roots and suffixes in Turkish.
- **CountVectorizer, LabelEncoder, Tokenizer:** To convert text data into numerical representations.
- **Keras Tuner:** For hyperparameter optimization.

The development environment used is **PyCharm 2021.2.3** with **Python 3.12**.
