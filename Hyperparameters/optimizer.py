import tensorflow as tf

lr_schedule = {
    'none' : lambda epoch,lr: lr,
}

optimizers = {
        'Adam' : lambda lr: tf.keras.optimizers.Adam(learning_rate=lr),
        'AdamW' : lambda lr: tf.keras.optimizers.AdamW(learning_rate=lr),
        'Nesterov' : lambda lr: tf.keras.optimizers.SGD(learning_rate=lr,nesterov=True),
}
