import numpy as np
import tensorflow as tf
import os

def verify_tflite(model_path='bottle_model.tflite'):
    print(f"Verifying TFLite model: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found.")
        return

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"Output shape: {output_data.shape}")
    print(f"Output sample (first 5 values): {output_data.flatten()[:5]}")

    if output_data.shape == (1, 27):
        print("SUCCESS: TFLite model verification passed!")
    else:
        print(f"FAILURE: Unexpected output shape {output_data.shape}")

if __name__ == "__main__":
    verify_tflite()
