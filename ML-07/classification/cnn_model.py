from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten

def build_cnn_model(config_type=1, input_dim=None):
    model = Sequential()
    
    if config_type == 1:
        # Config 1: มี 1 Convolutional Layer
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(input_dim, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
    elif config_type == 2:
        # Config 2: มี 2 Convolutional Layers เพื่อเปรียบเทียบ
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(input_dim, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model