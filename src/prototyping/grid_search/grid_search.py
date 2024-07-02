import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention

def transformer_encoder(inputs, num_heads, d_model, dff, rate):
    # Attention block
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attn_output = Dropout(rate)(attn_output)
    attn_output = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    
    # Feedforward block
    ffn_output = Dense(dff, activation='relu')(attn_output)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(rate)(ffn_output)
    ffn_output = LayerNormalization(epsilon=1e-6)(attn_output + ffn_output)
    return ffn_output

def build_model(input_shape, num_encoder_blocks, num_heads, d_model, dff, rate, positional_encoding):
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Positional Encoding (if enabled)
    if positional_encoding:
        # Ensuring positional encoding has the same dimension as d_model
        position = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        position_embedding = Dense(d_model, use_bias=False)(tf.expand_dims(position, 0))
        x += position_embedding

    # Adding encoder blocks
    for _ in range(num_encoder_blocks):
        x = transformer_encoder(x, num_heads=num_heads, d_model=d_model, dff=dff, rate=rate)

    # Example final layer for classification
    x = Dense(10, activation='softmax')(x[:, 0, :])  # Assuming classifying based on the first token
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class CustomModel(BaseEstimator, ClassifierMixin):  
    def __init__(self, input_shape=(10, 200), num_encoder_blocks=1, num_heads=1, d_model=25, dff=32, rate=0.1, positional_encoding=True, batch_size=32):
        self.input_shape = input_shape
        self.num_encoder_blocks = num_encoder_blocks
        self.num_heads = num_heads
        self.d_model = d_model
        self.dff = dff
        self.rate = rate
        self.positional_encoding = positional_encoding
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        self.model = self._build_model()
        self.model.fit(X, y, batch_size=self.batch_size, epochs=10, verbose=1)
        return self

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]

    def _build_model(self):
        return build_model(self.input_shape, self.num_encoder_blocks, self.num_heads, self.d_model, self.dff, self.rate, self.positional_encoding)

# Sample synthetic dataset for demonstration
num_samples = 1000
X = np.random.random((num_samples, 10, 200)).astype(np.float32)  # Adjust shape to match input_shape in param_grid
y = np.random.randint(0, 10, size=(num_samples,))

# Parameter grid for GridSearchCV
# param_grid = {
#     'num_encoder_blocks': [1, 8],
#     'num_heads': [1, 8],
#     'd_model': [200, 400],  # Ensured d_model matches or transforms correctly to input feature size
#     'dff': [32, 2048],
#     'rate': [0.1, 0.4],
#     'positional_encoding': [True, False],
#     'batch_size': [32, 128]
# }
param_grid = {
    'num_encoder_blocks': [1],
    'num_heads': [1],
    'd_model': [200],  # Ensured d_model matches or transforms correctly to input feature size
    'dff': [32],
    'rate': [0.1],
    'positional_encoding': [True],
    'batch_size': [32]
}

model = CustomModel(input_shape=(10, 200))  # Model initialized with one of the shapes
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X, y)  # Fitting on the synthetic dataset

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Print detailed results for each parameter configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
results = grid_result.cv_results_
for i in range(len(results['mean_test_accuracy'])):
    print(f"Configuration {i+1}:")
    print(f"Params: {results['params'][i]}")
    print(f"Accuracy: {results['mean_test_accuracy'][i]:.3f} (+/- {results['std_test_accuracy'][i]:.3f})")
    print(f"Precision: {results['mean_test_precision'][i]:.3f}")
    print(f"Recall: {results['mean_test_recall'][i]:.3f}")
    print(f"F1-score: {results['mean_test_f1'][i]:.3f}")
