import minitensor as mt
from minitensor.layers import Linear
from minitensor.losses import MSE
from minitensor.optims import SGD

learning_rate = 0.0001 #no normalization is performed

layer1 = Linear(input_features=1, output_features=1, dtype='float32')

X_train = mt.tensor([i for i in range(100)], shape=[100, 1], dtype='float32')
y_train = mt.tensor([float(i * 2) for i in range(100)], shape=[100, 1], dtype='float32')

optimizer = SGD(list(layer1.parameters()), lr=learning_rate)
loss_fn = MSE()

for i in range(100):
    optimizer.zero_grad()

    predictions = layer1(X_train)

    loss = loss_fn(y_train, predictions)

    loss.backward()

    optimizer.step()

    if i % 10 == 0:
        print(f"Epoch {i}, Loss: {loss.nested[0]}")

test_input = mt.tensor([101.0], shape=[1,1], dtype='float32')
predicted_output = layer1(test_input)
print(f"\nx: 101.0")
print(f"y_hat: {predicted_output.nested[0]}")
print(f"real y: 202.0")