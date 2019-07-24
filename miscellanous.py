

mag_accuracies = []
parity_accuracies = []

lrs = [0.01, 0.005, 0.001, 0.0005, 0.00001, 0.00005]

for rate in lrs:
    
    model1 = Model()

    inputs = Input(shape=(784,))

    layer1 = Dense(100, activation='tanh')(inputs)
    layer2 = Dense(100, activation='tanh')(layer1)

    parity_output = Dense(2, activation='softmax',name='parity_output')(layer2)
    magnitude_output = Dense(2, activation='softmax', name='magnitude_output')(layer2)


    model1 = Model(inputs=inputs, outputs=[parity_output, magnitude_output])


    model1.compile(keras.optimizers.Adam(lr=rate), loss={'parity_output': 'categorical_crossentropy', 'magnitude_output': 'categorical_crossentropy'})


    
    model1.fit(train_image_sample, [parity_train_sample, magnitude_train_sample], epochs=400)
    
    predictions = model1.predict(test_image_sample)
    
    accuracy = get_accuracy(predictions)
    
    mag_accuracies.append(accuracy[0])
    print(accuracy[0])
    
    parity_accuracies.append(accuracy[1])
    print(accuracy[1])


    
