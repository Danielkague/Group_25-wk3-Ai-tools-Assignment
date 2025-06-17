# Part 1: Theoretical Understanding

[Ai-Tools theoretical flamework.pdf](https://github.com/user-attachments/files/20738046/Ai-Tools.theoretical.flamework.pdf)

# Part 2: Practical Implementation

# Task 1: Classical ML with Scikit-learn

# Iris Flower Species Classification

## What This Program Does

This program teaches a computer to automatically identify different species of iris flowers based on their physical measurements. It's like training a digital botanist that can look at flower measurements and tell you exactly which type of iris it is!

## The Problem We're Solving

Imagine you're a botanist who has found an iris flower, but you're not sure which of the three main species it belongs to:
- **Setosa** - typically has shorter, wider petals
- **Versicolor** - has medium-sized features  
- **Virginica** - usually has longer, narrower petals

Instead of guessing or spending hours with identification guides, this program can instantly tell you the species based on just four simple measurements.

## How It Works (In Simple Terms)

### 1. **The Training Data**
The program starts with a famous dataset of 150 iris flowers where botanists have already:
- Measured 4 features of each flower:
  - Sepal length (the green leaf-like parts)
  - Sepal width
  - Petal length (the colorful flower parts)
  - Petal width
- Correctly identified which species each flower belongs to

### 2. **The Learning Process**
The program uses something called a "Decision Tree" - think of it like a flowchart of yes/no questions:
- "Is the petal length less than 2.5 cm?" → If yes, it's probably Setosa
- "Is the petal width greater than 1.7 cm?" → If yes, it might be Virginica
- And so on...

The computer automatically figures out the best questions to ask by looking at the training data.

### 3. **Making Predictions**
Once trained, you can give the program measurements of a new iris flower, and it will follow its decision tree to predict the species.

## What The Program Accomplishes

### 📊 **High Accuracy**
- Achieves over 90% accuracy in identifying iris species
- This means out of 100 iris flowers, it correctly identifies more than 90 of them

### 🔍 **Detailed Analysis**
The program provides:
- **Feature Importance**: Which measurements are most helpful for identification (usually petal measurements are more important than sepal measurements)
- **Confidence Scores**: How certain the program is about each prediction
- **Error Analysis**: What mistakes it makes and why

### 📈 **Visual Results**
Creates easy-to-understand charts showing:
- Which flower measurements matter most
- How often it makes correct vs incorrect predictions
- The actual decision-making process as a visual tree

## Real-World Applications

This same approach can be used for:
- **Medical Diagnosis**: Identifying diseases based on symptoms
- **Quality Control**: Detecting defective products in manufacturing
- **Image Recognition**: Identifying objects in photos
- **Financial Analysis**: Predicting market trends
- **Agriculture**: Classifying crops or detecting plant diseases

## How to Use This Program

### Prerequisites
You need to have Python installed on your computer with these libraries:
- pandas (for handling data)
- scikit-learn (for machine learning)
- matplotlib (for creating charts)
- seaborn (for beautiful visualizations)

### Running the Program
1. Save the code to a file called `classical_ML_sci_kitlearn.ipynb`
2. Open your command line or terminal
3. Navigate to the folder containing the file
4. Click: `Run All`
5. Watch as the program trains itself and shows you the results!

## Understanding the Output

When you run the program, you'll see:

### Step 1: Data Loading
- Shows you the dataset size and what information is available
- Displays the first few flower records

### Step 2: Data Preparation
- Checks if any measurements are missing
- Prepares the data for training

### Step 3: Training
- The computer learns patterns from the training data
- Shows which measurements are most important for classification

### Step 4: Testing Performance
- Tests the trained model on flowers it hasn't seen before
- Reports accuracy, precision, and recall scores
- Creates a "confusion matrix" showing exactly what mistakes were made

### Step 5: Visualizations
- Charts showing feature importance
- Heat map of prediction accuracy
- Visual representation of the decision tree

## Key Metrics Explained

- **Accuracy**: Percentage of correct predictions (higher is better)
- **Precision**: When the program says "this is species X," how often is it right?
- **Recall**: Of all flowers that are actually species X, how many did the program correctly identify?
- **Confusion Matrix**: A table showing exactly which species get confused with which others

# Task 2: Deep Learning with PyTorch

## MNIST Handwritten Digits Classification with CNN

A complete implementation of a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using Pytorch. This project achieves >95% accuracy on the test set and includes comprehensive visualization and analysis tools.

## Overview

This project implements a deep learning solution for recognizing handwritten digits (0-9) using a Convolutional Neural Network. The MNIST dataset contains 70,000 grayscale images of handwritten digits, each 28x28 pixels in size.

## Features

- **High-Performance CNN Architecture**: Multi-layer convolutional network with batch normalization and dropout
- **Comprehensive Evaluation**: Detailed performance metrics, confusion matrix, and per-digit analysis
- **Visual Analysis**: Training history plots, prediction visualization, and sample image analysis
- **Google Colab Ready**: Optimized for cloud-based training with GPU support
- **Robust Training**: Includes callbacks for early stopping and learning rate scheduling

## Model Architecture

The CNN consists of three main components:

### 1. Convolutional Blocks
- **Block 1**: 2x Conv2D (32 filters) + BatchNorm + MaxPooling + Dropout
- **Block 2**: 2x Conv2D (64 filters) + BatchNorm + MaxPooling + Dropout  
- **Block 3**: 1x Conv2D (128 filters) + BatchNorm + Dropout

### 2. Fully Connected Layers
- Flatten layer to convert 2D feature maps to 1D
- Dense layer (512 neurons) with ReLU activation
- Dense layer (256 neurons) with ReLU activation
- Output layer (10 neurons) with softmax activation

### 3. Regularization Techniques
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout**: Prevents overfitting (rates: 0.25 for conv layers, 0.5 for dense layers)
- **Early Stopping**: Prevents overtraining based on validation accuracy
- **Learning Rate Reduction**: Adaptive learning rate scheduling

## Requirements

```torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.5
matplotlib>=3.4.3
scikit-learn>=0.24.2
seaborn>=0.11.2
streamlit>=1.22.0
Pillow>=9.0.0 
opencv-python>=4.5.3
```

## How It Works

### 1. Data Preprocessing
```python
# Load MNIST dataset (60k training, 10k test images)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values from [0, 255] to [0, 1]
x_train = x_train.astype('float32') / 255.0

# Reshape for CNN: (samples, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoding
y_train_cat = keras.utils.to_categorical(y_train, 10)
```

### 2. Model Training
The model uses:
- **Adam Optimizer**: Adaptive learning rate optimization
- **Categorical Crossentropy Loss**: Suitable for multi-class classification
- **Accuracy Metric**: Primary evaluation metric
- **Batch Size**: 128 samples per training step
- **Epochs**: Up to 5 (with early stopping)

### 3. Performance Evaluation
The code provides multiple evaluation methods:
- **Overall Accuracy**: Test set performance
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: Visual representation of classification errors
- **Per-Digit Analysis**: Individual digit recognition accuracy

## Key Functions

### `create_cnn_model()`
Builds the CNN architecture with optimized hyperparameters for MNIST classification.

### `plot_training_history(history)`
Visualizes training and validation accuracy/loss curves to monitor model performance.

### `visualize_predictions(model, x_test, y_test, num_samples=5)`
Shows sample predictions with confidence scores and correctness indicators.

### `analyze_per_digit_performance(y_true, y_pred)`
Analyzes model performance for each digit (0-9) individually.

## Usage

1. **Run in Google Colab or Jupyter Notebook**:
   Simply execute all cells in sequence. The code is designed to run end-to-end.

2. **Expected Output**:
   - Model training progress with accuracy/loss metrics
   - Test accuracy >95% (typically 98-99%)
   - Training history plots
   - Confusion matrix visualization
   - Sample prediction visualizations
   - Per-digit performance analysis

3. **Model Saving**:
   The trained model is automatically saved as `mnist_cnn_model.h5`

## Performance Expectations

- **Test Accuracy**: >95% (typically 98-99%)
- **Training Time**: 2-5 minutes (depending on hardware)
- **Model Size**: ~2.5M parameters
- **Memory Usage**: <1GB RAM

## Understanding the Results

### Training History Plots
- **Accuracy Plot**: Shows model learning progression
- **Loss Plot**: Indicates convergence and potential overfitting

### Confusion Matrix
- Diagonal elements: Correct predictions
- Off-diagonal elements: Misclassifications
- Darker colors indicate higher frequencies

### Per-Digit Analysis
- Identifies which digits are easiest/hardest to classify
- Typically, digits 1 and 0 have highest accuracy
- Digits 8 and 9 often have lower accuracy due to similarity

## Technical Details

### Why This Architecture Works
1. **Convolutional Layers**: Extract spatial features (edges, shapes, patterns)
2. **Multiple Filters**: Capture different types of features at each level
3. **Pooling**: Reduces spatial dimensions while retaining important features
4. **Batch Normalization**: Accelerates training and improves stability
5. **Dropout**: Prevents overfitting by randomly disabling neurons during training

### Hyperparameter Choices
- **Filter Sizes**: 3x3 kernels balance feature extraction and computational efficiency
- **Number of Filters**: Progressive increase (32→64→128) captures increasingly complex features
- **Dropout Rates**: Lower for conv layers (0.25), higher for dense layers (0.5)
- **Dense Layer Sizes**: 512→256 provides sufficient capacity without overfitting

## Troubleshooting

### Common Issues
1. **Low Accuracy (<95%)**: Increase epochs or adjust learning rate
2. **Overfitting**: Increase dropout rates or add more regularization
3. **Memory Errors**: Reduce batch size or model complexity
4. **Slow Training**: Ensure GPU is available and properly configured

### Performance Tips
- Use GPU acceleration when available
- Monitor validation accuracy to prevent overfitting
- Experiment with different architectures for improved performance

## Extensions

Potential improvements and modifications:
- **Data Augmentation**: Rotation, shifting, scaling for better generalization
- **Ensemble Methods**: Combine multiple models for higher accuracy
- **Transfer Learning**: Use pre-trained features for faster training
- **Hyperparameter Tuning**: Systematic optimization of model parameters

# Task 3: NLP with spaCy

# Amazon Review Analyzer

A Python-based Natural Language Processing (NLP) tool that analyzes Amazon product reviews to extract sentiment, identify brands and products, and provide comprehensive insights using spaCy.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Output Examples](#output-examples)
- [Code Structure](#code-structure)
- [Customization](#customization)

## Overview

This tool processes Amazon product reviews to:
- **Analyze sentiment** (positive, negative, neutral) using rule-based classification
- **Extract entities** like brands, products, organizations, and monetary values
- **Identify products** through pattern matching and named entity recognition
- **Generate summary statistics** across multiple reviews

## Features

### 🎯 Sentiment Analysis
- Rule-based sentiment classification using predefined word dictionaries
- Handles negations (e.g., "not recommend" → negative sentiment)
- Confidence scoring based on sentiment word density
- Support for compound phrases like "highly recommend"

### 🏷️ Named Entity Recognition (NER)
- **Brands**: Detects major brands (Apple, Samsung, Nike, etc.)
- **Products**: Identifies product names with model numbers
- **Organizations**: Extracts company mentions
- **Money**: Finds price references
- **Locations**: Identifies geographical entities

### 📊 Analysis Features
- Individual review analysis with detailed breakdowns
- Batch processing of multiple reviews
- Summary statistics and trending insights
- Brand and product mention frequency tracking

## Installation

### Prerequisites
```bash
# Install required packages
pip install spacy pandas

# Download spaCy English language model
python -m spacy download en_core_web_sm
```

### Dependencies
- `spacy` - For NLP processing and entity recognition
- `pandas` - For data handling (imported but not actively used in current version)
- `collections.Counter` - For frequency counting
- `re` - For regex pattern matching

## How It Works

### 1. Sentiment Analysis Engine

The sentiment analyzer uses a **rule-based approach** with predefined word dictionaries:

**Positive Words**: love, amazing, excellent, fantastic, great, wonderful, perfect, etc.
**Negative Words**: hate, terrible, awful, horrible, bad, poor, worst, disappointing, etc.

#### Sentiment Calculation Process:
1. **Tokenization**: Text is split into individual words using spaCy
2. **Word Matching**: Each word is checked against positive/negative dictionaries
3. **Phrase Detection**: Special handling for phrases like "highly recommend" or "not recommend"
4. **Negation Handling**: Detects negation words and flips sentiment of following positive words
5. **Confidence Scoring**: Calculated as `sentiment_words / total_sentiment_words`

#### Example:
```
"I love this product but it's not perfect"
→ Positive: 2 (love, perfect), Negative: 0
→ Negation detected: "not perfect" → Positive: 1, Negative: 1
→ Result: NEUTRAL (50% confidence)
```

### 2. Named Entity Recognition (NER)

The system combines **spaCy's built-in NER** with **custom pattern matching**:

#### Standard spaCy Entities:
- `PERSON`: People's names
- `ORG`: Organizations and companies
- `PRODUCT`: Product mentions
- `MONEY`: Monetary values
- `GPE`: Countries, cities, states

#### Custom Brand Detection:
- Predefined list of major brands (Apple, Samsung, Sony, Nike, etc.)
- Case-insensitive matching within review text
- Deduplication to avoid multiple mentions of the same brand

#### Product Extraction:
Uses regex pattern: `\b[A-Z][a-zA-Z0-9\s\-]{2,30}\b`
- Identifies capitalized phrases (potential product names)
- Filters for products containing:
  - Model numbers (iPhone 14, Galaxy S23)
  - Product keywords (Pro, Max, Plus, Air, Mini, Ultra)

### 3. Analysis Workflow

```
Input Review Text
       ↓
   spaCy Processing (tokenization, POS tagging, NER)
       ↓
   ┌─────────────────┬─────────────────┐
   ↓                 ↓                 ↓
Sentiment Analysis  Entity Extraction  Custom Pattern Matching
       ↓                 ↓                 ↓
   Word Scoring      Standard NER      Brand Detection
   Negation Check    Product Regex     Product Filtering
       ↓                 ↓                 ↓
   ┌─────────────────┬─────────────────┐
   ↓                                   ↓
Combined Results                   Summary Statistics
```

## Usage

### Basic Usage

```python
from amazon_review_analyzer import AmazonReviewAnalyzer

# Initialize the analyzer
analyzer = AmazonReviewAnalyzer()

# Analyze a single review
review = "I love my new iPhone 14! Apple has outdone themselves."
result = analyzer.analyze_review(review)

print(f"Sentiment: {result['sentiment']['sentiment']}")
print(f"Brands found: {result['entities']['BRANDS']}")
print(f"Products found: {result['entities']['PRODUCT']}")
```

### Batch Analysis

```python
# Analyze multiple reviews
reviews = [
    "Amazing Nike sneakers! Highly recommend.",
    "Terrible Samsung phone. Screen issues from day one.",
    "The Canon camera is decent but overpriced."
]

results = analyzer.analyze_multiple_reviews(reviews)
```

### Running the Example with Jupyter notebooks/ Google colab

Navigate to NLP_with_spaCy.ipynb then click Run All



This will run the analysis on the included sample reviews and display detailed results.

## Output Examples

### Individual Review Analysis
```
REVIEW 1:
Text: I absolutely love my new iPhone 14! Apple has outdone themselves...
Sentiment: POSITIVE (Confidence: 0.75)
Brands: Apple
Products: iPhone 14
Organizations: Apple
------------------------------------------------------------
```

### Summary Statistics
```
=== ANALYSIS SUMMARY ===
Total Reviews Analyzed: 10

Sentiment Distribution:
  POSITIVE: 6 (60.0%)
  NEGATIVE: 3 (30.0%)
  NEUTRAL: 1 (10.0%)

Most Mentioned Brands:
  Apple: 4 mentions
  Samsung: 2 mentions
  Nike: 1 mentions

Most Mentioned Products:
  iPhone 14: 2 mentions
  MacBook Pro: 1 mentions
  Galaxy S23: 1 mentions
```

## Code Structure

### Main Class: `AmazonReviewAnalyzer`

#### Key Methods:

**`__init__()`**
- Initializes spaCy model
- Sets up positive/negative word dictionaries
- Defines brand patterns for detection

**`extract_entities(text)`**
- Processes text with spaCy NLP pipeline
- Extracts standard named entities
- Applies custom brand detection
- Uses regex for product name extraction
- Returns structured entity dictionary

**`analyze_sentiment(text)`**
- Tokenizes text and counts sentiment words
- Handles multi-word phrases and negations
- Calculates confidence scores
- Returns sentiment classification with metrics

**`analyze_review(review_text)`**
- Combines entity extraction and sentiment analysis
- Returns comprehensive analysis results

**`analyze_multiple_reviews(reviews)`**
- Processes multiple reviews in batch
- Generates summary statistics
- Displays formatted results
- Returns aggregated insights

### Data Structures

#### Sentiment Result:
```python
{
    'sentiment': 'POSITIVE',      # POSITIVE, NEGATIVE, or NEUTRAL
    'confidence': 0.75,           # Confidence score (0-1)
    'positive_score': 3,          # Count of positive words
    'negative_score': 1           # Count of negative words
}
```

#### Entity Result:
```python
{
    'PERSON': ['John Smith'],
    'ORG': ['Apple Inc'],
    'PRODUCT': ['iPhone 14 Pro'],
    'BRANDS': ['Apple', 'Samsung'],
    'MONEY': ['$999'],
    'GPE': ['United States']
}
```

## Customization

### Adding New Sentiment Words
```python
# Extend positive/negative word dictionaries
analyzer.positive_words.update({'phenomenal', 'outstanding', 'brilliant'})
analyzer.negative_words.update({'catastrophic', 'abysmal', 'dreadful'})
```

### Adding New Brands
```python
# Add more brands to detection patterns
analyzer.brand_patterns.update({'Tesla', 'SpaceX', 'Netflix'})
```

### Adjusting Product Detection
Modify the regex pattern in `extract_entities()` to catch different product naming conventions:
```python
# Current pattern: \b[A-Z][a-zA-Z0-9\s\-]{2,30}\b
# Custom pattern for your use case
product_pattern = r'\b[A-Z][a-zA-Z0-9\s\-\.]{3,40}\b'
```

### Enhancing Sentiment Analysis
- Add industry-specific sentiment words
- Implement weighted scoring for different word types
- Add context-aware sentiment analysis
- Integrate with machine learning models

## Limitations & Future Improvements

### Current Limitations:
- Rule-based sentiment analysis (not ML-based)
- Limited to predefined brand list
- Simple negation handling
- English language only

### Potential Enhancements:
- Machine learning sentiment classification
- Aspect-based sentiment analysis
- Multi-language support
- Integration with review platforms' APIs
- Real-time analysis capabilities
- Export to CSV/JSON formats

## Contributing

To contribute to this project:
1. Fork the repository
2. Add new features or improve existing ones
3. Update documentation and tests
4. Submit a pull request
   
### Part 3: Ethics & Optimization (10%)

### 1. Ethical Considerations

When developing machine learning models using datasets like MNIST or Amazon Reviews, there are several potential ethical biases to consider:

### A. Potential Biases in MNIST or Amazon Reviews Models
### MNIST (Handwritten Digits Dataset)
Dataset Bias: MNIST primarily contains digits written by US Census Bureau employees in the 1990s. This could cause:

Poor generalization to other handwriting styles (e.g., children, people from non-Western cultures).

Lower performance for digits written with different writing instruments or cultural formats.

### Amazon Reviews (Text Sentiment Dataset)
Demographic Bias: Reviews may disproportionately reflect the language and opinions of specific demographic groups (e.g., English speakers from the U.S.).

Sentiment Bias: Language used by certain groups (e.g., African-American Vernacular English or non-native English speakers) might be incorrectly interpreted as negative or neutral due to biased training data.

Product Category Bias: Some product types may receive more or fewer positive reviews due to cultural preferences or gender norms, skewing model predictions.

### B. Mitigating Biases Using Tools
### 1. TensorFlow Fairness Indicators
These indicators evaluate performance across subgroups (e.g., gender, age, language style) and provide visual dashboards.

Use case: In Amazon Reviews, you could analyze accuracy, precision, and false positive rates for different user demographics or writing styles to spot unfair behavior.

Example Fix: If the model underperforms on reviews with informal or regional language, retrain the model with a more representative dataset or apply data augmentation.

### 2. spaCy’s Rule-Based Systems
spaCy allows integration of custom rule-based NLP components, which can:

Detect and flag potentially biased or sensitive language during preprocessing.

Normalize slang or dialect terms to reduce misclassification risk.

Use case: If certain terms from a minority dialect are associated with negative sentiment due to lack of training data, spaCy can be used to re-label or re-categorize them fairly.

### Troubleshooting Challenge
Troubleshooting refers to the process of identifying, diagnosing, and resolving problems in systems or processes. It also refers to a challenge that can be defined as a task or situation that requires the identification and correction of faults in a malfunctioning system. 

### Debug and fix pytorch code 
1️⃣ Classification Tasks → Use Classification Losses
Binary Classification (e.g., spam vs. not spam): Use Binary Cross-Entropy Loss (also called Log Loss)

Multi-Class Classification (e.g., digit recognition 0–9): Use Categorical Cross-Entropy Loss

Multi-Label Classification (e.g., tagging an image with multiple labels): Use Binary Cross-Entropy per label with sigmoid activation

2️⃣ Regression Tasks → Use Regression Losses
Standard Regression (e.g., predicting house prices): Use Mean Squared Error (MSE) or Mean Absolute Error (MAE)

Robust Regression (to reduce impact of outliers): Try Huber Loss or Quantile Loss

3️⃣ Ranking or Structured Outputs
Ranking problems (e.g., search engine results): Use Hinge Loss or Pairwise Ranking Loss

Sequence generation tasks (e.g., language translation): Often use Cross-Entropy, sometimes with teacher forcing and BLEU score as an eval metric

### Conclusion 

By using tools like pytorch Fairness Indicators for subgroup evaluation and spaCy's rule-based systems for preprocessing, developers can detect, analyze, and address unfair bias—creating models that are not only accurate but also ethically responsible.







