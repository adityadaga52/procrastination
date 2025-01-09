# Neural Network for Procrastination Evaluation

This project implements a neural network to analyze procrastination patterns among students. It uses training data on student attempts, assignment grades, and completion times to predict the likelihood of passing or failing a course based on features like submission times and grades.

## Features
The neural network evaluates the following features (configurable):
- Start time of the first attempt
- End time of the last attempt
- Grade at the start and end of the attempts
- Maximum grade achieved
- Total number of attempts

## Key Functionalities
1. **Data Preprocessing**:
   - Parses data from CSV files for student attempts and grades.
   - Creates dictionaries for attempts and grades.
   - Generates feature vectors for each student.

2. **Neural Network**:
   - Uses TensorFlow and Keras to build a fully connected feed-forward neural network.
   - Supports 10-fold cross-validation for model evaluation.
   - Tracks training accuracy, precision, and recall.

3. **Reporting**:
   - Outputs overall training metrics.
   - Provides individual and average results for 10-fold cross-validation.

## Project Structure
- **Data Inputs**:
  - `attempts.csv`: Contains data on student attempts (student ID, assignment, attempt, grade, time).
  - `grades.csv`: Contains final grade data for students (student ID, grade).
- **Code Workflow**:
  - Data processing: Parses and preprocesses data.
  - Feature extraction: Generates input vectors for the neural network.
  - Model training: Conducts 10-fold cross-validation.
  - Results reporting: Outputs training metrics and averages.

## Requirements
Install the following dependencies:
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

Install the packages using pip:
```bash
pip install tensorflow numpy matplotlib
