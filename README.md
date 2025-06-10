# Code Plagiarism Detection System

A machine learning-based system for detecting code plagiarism using Abstract Syntax Tree (AST) analysis and neural networks. This tool extracts structural features from Python code and uses both traditional similarity metrics and deep learning to identify potentially plagiarized code snippets.

## Features

- **AST-based Feature Extraction**: Analyzes code structure rather than surface-level text
- **TF-IDF Vectorization**: Converts code into numerical vectors for comparison
- **Neural Network Classification**: Deep learning model for plagiarism detection
- **Cosine Similarity**: Traditional similarity metric for baseline comparison
- **Synthetic Data Generation**: Automated training data creation
- **GPU Support**: CUDA acceleration when available

## Architecture Overview

### 1. AST Feature Extractor
The `ASTFeatureExtractor` class converts Python code into numerical vectors by:
- Parsing code into Abstract Syntax Trees
- Counting occurrences of specific AST node types
- Computing TF-IDF (Term Frequency-Inverse Document Frequency) scores
- Generating fixed-size feature vectors

### 2. Neural Network Model
The `PlagiarismDetector` is a feed-forward neural network with:
- Input layer matching feature vector dimensions
- Three hidden layers (128, 64, 32 neurons)
- ReLU activation functions
- Dropout layers for regularization
- Sigmoid output for binary classification

### 3. Training Pipeline
- Synthetic data generation with similar/dissimilar code pairs
- Batch processing with PyTorch DataLoader
- Adam optimizer with binary cross-entropy loss
- Configurable training epochs and batch sizes

## Installation

### Prerequisites
```bash
pip install torch numpy ast collections math
```

### Dependencies
- **PyTorch**: Neural network framework
- **NumPy**: Numerical computations
- **Python AST**: Built-in Abstract Syntax Tree module
- **Collections**: Counter and defaultdict utilities
- **Math**: Mathematical operations

## Usage

### Basic Usage

```python
from main import ASTFeatureExtractor, cosine_similarity

# Initialize the feature extractor
extractor = ASTFeatureExtractor()

# Extract features from code snippets
code1 = """
def addition(a, b):
    return a + b
print(addition(1, 2))
"""

code2 = """
def multiply(x, y):
    return x * y
result = multiply(3, 4)
"""

# Get feature vectors
vector1 = extractor.extract_features(code1)
vector2 = extractor.extract_features(code2)

# Calculate similarity
similarity = cosine_similarity(vector1, vector2)
print(f"Similarity: {similarity:.4f}")
```

### Training the Neural Network

```python
from main import main

# Run the complete training pipeline
main()
```

This will:
1. Generate synthetic training data
2. Train the neural network model
3. Test the model with example code snippets
4. Display both neural network predictions and cosine similarity scores

### Advanced Usage

#### Custom Feature Extraction
```python
# Create extractor with custom node types
extractor = ASTFeatureExtractor()
extractor.target_types = ['FunctionDef', 'Return', 'BinOp']  # Custom types

# Extract features
features = extractor.extract_features(your_code)
```

#### Model Training Parameters
```python
# Customize training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlagiarismDetector(input_dim=25).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train with custom settings
train_model(model, train_loader, criterion, optimizer, device, epochs=20)
```

## Code Structure

### Core Classes

#### `ASTFeatureExtractor`
- **Purpose**: Converts Python code to numerical feature vectors
- **Key Methods**:
  - `extract_features(code_string)`: Main feature extraction method
  - `_get_node_frequencies(tree)`: Counts AST node occurrences
  - `_calculate_tfidf_vector(doc_freq)`: Computes TF-IDF scores

#### `PlagiarismDetector`
- **Purpose**: Neural network for plagiarism classification
- **Architecture**: 4-layer feedforward network with dropout
- **Input**: Feature vectors from AST analysis
- **Output**: Plagiarism probability (0-1)

#### `CodeDataset`
- **Purpose**: PyTorch dataset wrapper for code vectors
- **Features**: Handles vector-label pairs for training

### Utility Functions

#### `cosine_similarity(vec1, vec2)`
Calculates cosine similarity between two feature vectors.

#### `train_model(...)`
Handles the neural network training loop with loss computation and backpropagation.

#### `generate_training_data(...)`
Creates synthetic training data with similar and dissimilar code pairs.

## Feature Vector Composition

The system extracts 25 different AST node types:

| Category | Node Types |
|----------|------------|
| **Functions** | FunctionDef, arguments, arg, Return |
| **Operations** | BinOp, UnaryOp, BoolOp, Compare |
| **Data Types** | Num, Str, List, Tuple, Dict, Set |
| **Control Flow** | For, While, If |
| **References** | Name, Call, Attribute |
| **Indexing** | Subscript, Slice, Index, ExtSlice |

Each code snippet is represented as a 25-dimensional vector where each dimension corresponds to the TF-IDF score of a specific AST node type.

## Performance Metrics

The system provides two similarity measures:

1. **Cosine Similarity**: Traditional metric based on vector dot product
2. **Neural Network Score**: Learned similarity from training data

### Interpretation
- **Score > 0.7**: High similarity (potential plagiarism)
- **Score 0.3-0.7**: Moderate similarity (further investigation needed)
- **Score < 0.3**: Low similarity (likely original code)

## Limitations

1. **Language Specific**: Currently supports Python code only
2. **Structural Focus**: May miss semantic similarities with different structures
3. **Training Data**: Performance depends on quality of training examples
4. **Variable Renaming**: Highly sensitive to identifier changes
5. **Comment Blind**: Ignores code comments and documentation

## Future Enhancements

- **Multi-language Support**: Extend to Java, C++, JavaScript
- **Semantic Analysis**: Incorporate meaning-based comparisons
- **Real Dataset Training**: Use actual plagiarism cases for training
- **Web Interface**: Browser-based plagiarism checking tool
- **Batch Processing**: Handle multiple file comparisons
- **Performance Optimization**: Faster feature extraction algorithms

## Example Output

```
Analyzing first code snippet:

AST Structure:
Module(
  body=[
    FunctionDef(
      name='addition',
      args=arguments(args=[arg(arg='a'), arg(arg='b')]),
      body=[Return(value=BinOp(left=Name(id='a'), op=Add(), right=Name(id='b')))]
    ),
    Expr(value=Call(func=Name(id='print'), args=[Call(func=Name(id='addition'), args=[Num(n=1), Num(n=2)])]))
  ]
)

TF-IDF Vector for first snippet:
Vector shape: (25,)
Non-zero elements: 8
Vector values: [0.1386 0.1386 0.1386 0.1386 0.1386 0.2773 0.2773 0.2773 0 0 ...]

Cosine Similarity between snippets: 0.8234
Model prediction for code similarity: 0.7891
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/enhancement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{code_plagiarism_detector,
  title={Code Plagiarism Detection System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/code-plagiarism-detector}
}
```
