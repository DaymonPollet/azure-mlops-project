import argparse
import os
from glob import glob
import random
import tensorflow as tf
import numpy as np
import io # Added for BytesIO
import matplotlib.pyplot as plt # Explicitly import matplotlib here for clarity

# This time we will need our Tensorflow Keras libraries, as we will be working with the AI training now
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# This AzureML package will allow to log our metrics etc.
from azureml.core import Run

# Important to load in the utils as well!
from utils import * # Assuming utils.py is in the same directory or accessible

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_folder', type=str, dest='training_folder', help='training folder mounting point')
    parser.add_argument('--testing_folder', type=str, dest='testing_folder', help='testing folder mounting point')
    parser.add_argument('--output_folder', type=str, dest='output_folder', help='Output folder')
    parser.add_argument('--epochs', type=int, dest='epochs', help='The amount of Epochs to train')

    # --- NEW DYNAMIC INPUTS FOR HYPERPARAMETERS AND MODEL NAME ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--initial_learning_rate", type=float, default=0.01, help="Initial learning rate for optimizer.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--patience", type=int, default=11, help="Patience for early stopping.")
    parser.add_argument("--model_name", type=str, default="animal-cnn", help="Name for the saved AI model.")
    # --- END NEW DYNAMIC INPUTS ---

    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    training_folder = args.training_folder
    print('Training folder:', training_folder)

    testing_folder = args.testing_folder
    print('Testing folder:', testing_folder)

    output_folder = args.output_folder
    print('Output folder:', output_folder)

    MAX_EPOCHS = args.epochs

    # As we're mounting the training_folder and testing_folder onto the `/mnt/data` directories, we can load in the images by using glob.
    training_paths = glob(os.path.join(training_folder, "*.jpg"), recursive=True)
    testing_paths = glob(os.path.join(testing_folder, "*.jpg"), recursive=True)

    # --- Use args.seed instead of hardcoded SEED ---
    random.seed(args.seed)
    random.shuffle(training_paths)
    random.seed(args.seed) # Ensure testing_paths also uses the same seed
    random.shuffle(testing_paths)

    print("Training samples:", len(training_paths))
    print("Testing samples:", len(testing_paths))

    print(training_paths[:3]) # Examples
    print(testing_paths[:3]) # Examples

    # Parse to Features and Targets for both Training and Testing. Refer to the Utils package for more information
    X_train = getFeatures(training_paths)
    y_train = getTargets(training_paths)

    X_test = getFeatures(testing_paths)
    y_test = getTargets(testing_paths)

    print('Shapes:')
    print(X_train.shape)
    print(X_test.shape)
    print(len(y_train))
    print(len(y_test))

    # Make sure the data is one-hot-encoded
    LABELS, y_train, y_test = encodeLabels(y_train, y_test)
    print('One Hot Shapes:')

    print(y_train.shape)
    print(y_test.shape)

    # Create an output directory where our AI model will be saved to.
    # Everything inside the `outputs` directory will be logged and kept aside for later usage.
    # --- Use args.model_name for the model path ---
    model_path = os.path.join(output_folder, args.model_name)
    os.makedirs(model_path, exist_ok=True)


    # Save the best model, not the last
    # --- Use args.model_name for the filepath ---
    cb_save_best_model = keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_path, 'best_model.h5'), # Save as .h5 inside the model_name folder
                                                         monitor='val_loss',
                                                         save_best_only=True,
                                                         verbose=1)

    # Early stop when the val_los isn't improving for PATIENCE epochs
    # --- Use args.patience instead of hardcoded PATIENCE ---
    cb_early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=args.patience,
                                                  verbose=1,
                                                  restore_best_weights=True)

    # Reduce the Learning Rate when not learning more for 4 epochs.
    cb_reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(factor=.5, patience=4, verbose=1)

    # Define the Optimizer
    # --- Use args.initial_learning_rate instead of hardcoded INITIAL_LEARNING_RATE ---
    opt = tf.keras.optimizers.legacy.SGD(learning_rate=args.initial_learning_rate, decay=args.initial_learning_rate / MAX_EPOCHS)

    model = buildModel((64, 64, 3), 3) # Create the AI model as defined in the utils script.

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Construct & initialize the image data generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")


    # train the network
    history = model.fit(aug.flow(X_train, y_train, batch_size=args.batch_size),
                            validation_data=(X_test, y_test),
                            steps_per_epoch=len(X_train) // args.batch_size,
                            epochs=MAX_EPOCHS,
                            callbacks=[cb_save_best_model, cb_early_stop, cb_reduce_lr_on_plateau] )

    print("[INFO] evaluating network...")
    predictions = model.predict(X_test, batch_size=args.batch_size)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=['cats', 'dogs', 'panda']))

    cf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
    print(cf_matrix)

    # Logging metrics to Azure ML Run context
    run = Run.get_context()
    run.log("accuracy", history.history['accuracy'][-1])
    run.log("val_accuracy", history.history['val_accuracy'][-1])
    
    # Convert cf_matrix to a list of lists before logging
    run.log_confusion_matrix("confusion_matrix", cf_matrix.tolist()) 
    
    # --- FIX: Save matplotlib figure to BytesIO and then log ---
    fig = plot_confusion_matrix(cf_matrix, ['cats', 'dogs', 'panda'])
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0) # Rewind the buffer to the beginning
    run.log_image("confusion_matrix_plot", plot=fig) # Try logging directly first
    # If the above line still causes issues, use the BytesIO content:
    # run.log_image("confusion_matrix_plot_bytes", path=None, plot=None, description="Confusion Matrix Plot", image=buf.getvalue())
    plt.close(fig) # Close the figure to free up memory
    # --- End FIX ---

    np.save(os.path.join(output_folder, 'confusion_matrix.npy'), cf_matrix) 

    print("DONE TRAINING")


if __name__ == "__main__":
    main()
