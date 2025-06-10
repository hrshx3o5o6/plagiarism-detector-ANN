import ast
from collections import deque, Counter, defaultdict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ASTFeatureExtractor:
    def __init__(self):
        self.node_frequencies = Counter()
        self.document_frequencies = defaultdict(int)
        self.total_documents = 0
        # Expanded target types for better feature representation
        self.target_types = [
            'FunctionDef', 'arguments', 'arg', 'Return', 'BinOp', 'Name',
            'Call', 'Num', 'Str', 'List', 'Tuple', 'Dict', 'Set',
            'For', 'While', 'If', 'Compare', 'BoolOp', 'UnaryOp',
            'Attribute', 'Subscript', 'Slice', 'Index', 'ExtSlice'
        ]
        # Initialize vector dimension
        self.vector_dim = len(self.target_types)
        # Create node type to index mapping
        self.node_to_index = {node_type: idx for idx, node_type in enumerate(self.target_types)}
    
    def extract_features(self, code_string):
        tree = ast.parse(code_string)
        # Print AST structure
        print("\nAST Structure:")
        print(ast.dump(tree, indent=2))
        
        # Get node frequencies for this document
        doc_freq = self._get_node_frequencies(tree)
        # Update document frequencies
        for node_type in doc_freq:
            self.document_frequencies[node_type] += 1
        self.total_documents += 1
        # Calculate TF-IDF vector
        return self._calculate_tfidf_vector(doc_freq)
    
    def _get_node_frequencies(self, tree):
        queue = deque([tree])
        frequencies = Counter()
        
        while queue:
            node = queue.popleft()
            node_type = type(node).__name__
            
            if node_type in self.target_types:
                frequencies[node_type] += 1
            
            for child in ast.iter_child_nodes(node):
                queue.append(child)
        
        return frequencies
    
    def _calculate_tfidf_vector(self, doc_freq):
        # Initialize vector with zeros
        vector = np.zeros(self.vector_dim)
        total_nodes = sum(doc_freq.values())
        
        for node_type, freq in doc_freq.items():
            if node_type in self.node_to_index:
                # Calculate TF (term frequency)
                tf = freq / total_nodes if total_nodes > 0 else 0
                
                # Calculate IDF (inverse document frequency)
                df = self.document_frequencies[node_type]
                idf = math.log((self.total_documents + 1) / (df + 1)) + 1
                
                # Calculate TF-IDF and store in vector
                vector[self.node_to_index[node_type]] = tf * idf
        
        return vector

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

# Example usage
code = """
def addition(a, b):
    return a + b

print(addition(1, 2))
"""

print("Analyzing first code snippet:")
extractor = ASTFeatureExtractor()
vector1 = extractor.extract_features(code)

print("\nTF-IDF Vector for first snippet:")
print("Vector shape:", vector1.shape)
print("Non-zero elements:", np.count_nonzero(vector1))
print("Vector values:", vector1)

# Example of comparing two code snippets
code2 = """
def multiply(x, y):
    return x * y

result = multiply(3, 4)
"""

print("\nAnalyzing second code snippet:")
vector2 = extractor.extract_features(code2)

print("\nTF-IDF Vector for second snippet:")
print("Vector shape:", vector2.shape)
print("Non-zero elements:", np.count_nonzero(vector2))
print("Vector values:", vector2)

# Calculate similarity
similarity = cosine_similarity(vector1, vector2)
print("\nCosine Similarity between snippets:", similarity)

class CodeDataset(Dataset):
    def __init__(self, vectors, labels):
        self.vectors = torch.FloatTensor(vectors)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]

class PlagiarismDetector(nn.Module):
    def __init__(self, input_dim):
        super(PlagiarismDetector, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

def generate_training_data(extractor, num_samples=1000):
    """Generate synthetic training data"""
    vectors = []
    labels = []
    
    # Generate similar pairs
    for _ in range(num_samples // 2):
        # Create a base code
        base_code = f"""
def func_{_}(x, y):
    return x + y
print(func_{_}(1, 2))
"""
        # Create a similar code with minor modifications
        similar_code = f"""
def func_{_}_modified(a, b):
    return a + b
result = func_{_}_modified(1, 2)
"""
        vec1 = extractor.extract_features(base_code)
        vec2 = extractor.extract_features(similar_code)
        vectors.extend([vec1, vec2])
        labels.extend([1, 1])  # Similar pair
    
    # Generate dissimilar pairs
    for _ in range(num_samples // 2):
        # Create two different codes
        code1 = f"""
def func_{_}(x):
    return x * 2
"""
        code2 = f"""
def different_func_{_}(y):
    for i in range(y):
        print(i)
"""
        vec1 = extractor.extract_features(code1)
        vec2 = extractor.extract_features(code2)
        vectors.extend([vec1, vec2])
        labels.extend([0, 0])  # Dissimilar pair
    
    return np.array(vectors), np.array(labels)

def main():
    # Initialize feature extractor
    extractor = ASTFeatureExtractor()
    
    # Generate training data
    print("Generating training data...")
    vectors, labels = generate_training_data(extractor)
    
    # Create dataset and dataloader
    dataset = CodeDataset(vectors, labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlagiarismDetector(extractor.vector_dim).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("\nTraining the model...")
    train_model(model, train_loader, criterion, optimizer, device)
    
    # Test the model with example code
    print("\nTesting the model with example code...")
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
    
    vec1 = extractor.extract_features(code1)
    vec2 = extractor.extract_features(code2)
    
    # Convert to tensor and get prediction
    vec1_tensor = torch.FloatTensor(vec1).unsqueeze(0).to(device)
    vec2_tensor = torch.FloatTensor(vec2).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        similarity1 = model(vec1_tensor).item()
        similarity2 = model(vec2_tensor).item()
    
    print(f"\nModel prediction for code similarity: {similarity1:.4f}")
    print(f"Cosine similarity: {cosine_similarity(vec1, vec2):.4f}")

if __name__ == "__main__":
    main()