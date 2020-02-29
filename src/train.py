import os

from tensorflow import keras
from tensorflow_core.python.keras.utils.vis_utils import plot_model

# This file holds the implementation of the training and logging procedures.
# Note that the network and loss are implemented in separate files.
from data import Data

INITIAL_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.98
LEARNING_RATE_DECAY_EVERY_N_EPOCHS = 2


def some_loss_fnc(y_true, y_pred):
    pass


def scheduler(epoch, learning_rate):
    if epoch > 0:
        if epoch % LEARNING_RATE_DECAY_EVERY_N_EPOCHS == 0:
            learning_rate = learning_rate * LEARNING_RATE_DECAY
            print("Change learning rate to", "{0:.6f}".format(learning_rate))
    return learning_rate


def train_network(
    experiment_dir,
    tensorboard_dir,
    batch_size,
    num_batches_train,
    num_batches_valid,
    num_epochs,
    num_epochs_for_early_stopping,
    optimizer_clip_l2_norm_value,
    tasnet,
):

    tasnet_data_train = Data()
    tasnet_data_valid = Data()

    train_generator = tasnet_data_train.batch_generator(
        batch_size=batch_size, num_batches=num_batches_train
    )
    validation_generator = tasnet_data_valid.batch_generator(
        batch_size=batch_size, num_batches=num_batches_valid
    )

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=os.path.join(tensorboard_dir), update_freq="batch", write_graph=True
    )

    model_save_callbback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(experiment_dir, "state_epoch_{epoch}.h5"),
        save_freq="epoch",
        save_weights_only=True,
        load_weights_on_restart=True,
    )

    learning_rate_callback = keras.callbacks.LearningRateScheduler(scheduler)
    early_stopping_callback = keras.callbacks.EarlyStopping(patience=num_epochs_for_early_stopping)

    adam = keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE, clipnorm=optimizer_clip_l2_norm_value
    )
    tasnet.model.compile(loss=some_loss_fnc, optimizer=adam)

    plot_model(model=tasnet.model, to_file="architecture.png")

    history = tasnet.model.fit_generator(
        train_generator,
        epochs=num_epochs,
        steps_per_epoch=num_batches_train,
        validation_data=validation_generator,
        validation_steps=num_batches_valid,
        validation_freq=1,
        callbacks=[
            tensorboard_callback,
            model_save_callbback,
            learning_rate_callback,
            early_stopping_callback,
        ],
    )
    return history.history["val_loss"]
