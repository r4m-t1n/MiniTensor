import minitensor as mt
import numpy as np

model = mt.model.Sequential(
    mt.layers.Linear(input_features=4, output_features=3),
    mt.activations.Softmax()
)

input_data = mt.tensor([
    [1.2, 0.5, -0.8, 2.5],
    [-0.3, 1.8, 2.1, 0.1]
], dtype='float32')


output_probabilities = model(input_data)

print("probabilities:")
print(np.round(output_probabilities.nested, 4))

output_sums = output_probabilities.sum(1)

print(f"\nsum of probabilities of the first sample: {output_sums[0]:.4f}")
print(f"sum of probabilities of the second sample: {output_sums[1]:.4f}")

predictions = np.argmax(output_probabilities.nested, axis=1)
print(f"\npredicted class for sample one: {predictions[0]}")
print(f"predicted class for sample two: {predictions[1]}")