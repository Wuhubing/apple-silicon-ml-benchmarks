import tensorflow as tf
import coremltools as ct
import numpy as np
import time

def create_simple_model():
    """Create a simple CNN model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def main():
    try:
        # 1. Create simple model
        print("Creating model...")
        model = create_simple_model()
        
        # 2. Convert to Core ML model
        print("Converting to Core ML model...")
        mlmodel = ct.convert(
            model,
            inputs=[ct.ImageType(
                name="input_1",
                shape=(1, 224, 224, 3),
                scale=1/255.0
            )],
            compute_units=ct.ComputeUnit.ALL
        )
        
        # 3. Save model
        mlmodel.save("SimpleModel.mlmodel")
        print("Model saved as SimpleModel.mlmodel")
        
        # 4. Test inference
        print("Testing inference...")
        test_input = np.random.rand(1, 224, 224, 3)
        result = mlmodel.predict({"input_1": test_input})
        print("Inference successful!")
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()