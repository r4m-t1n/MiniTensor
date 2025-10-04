import minitensor as mt
from minitensor.layers import Linear
from minitensor.activations import Sigmoid
from minitensor.losses import BCE
from minitensor.optims import SGD

raw_X = [i for i in range(100)]
X_norm = [(x - min(raw_X)) / (max(raw_X) - min(raw_X)) for x in raw_X] # I'll implement a built-in normalizer soon
X_train = mt.tensor([[x] for x in X_norm], dtype='float32')
y_train = mt.tensor([[1.0 if x[0] >= 0.5 else 0.0] for x in X_train.nested], dtype='float32')

layer1 = Linear(input_features=1, output_features=1, dtype='float32')
activation = Sigmoid()

learning_rate = 0.0005
optimizer = SGD(list(layer1.parameters()), lr=learning_rate)
loss_fn = BCE()

for epoch in range(500):
    optimizer.zero_grad()
    logits = layer1(X_train)
    predictions = activation(logits)
    loss = loss_fn(y_train, predictions)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.nested[0]:.6f}")

test_values = [10, 45, 50, 70, 90]
test_norm = [(x - min(raw_X)) / (max(raw_X) - min(raw_X)) for x in test_values]
test_inputs = mt.tensor([[x] for x in test_norm], dtype='float32')
test_predictions = activation(layer1(test_inputs))

print("\nPredictions:")
for i, p in enumerate(test_predictions.nested):
    print(f"x = {test_values[i]}, predicted = {p[0]:.4f} ({'class 1' if p[0] >= 0.5 else 'class 0'})")
