# ============================================================
# U-Net for learning Q(z,y,t) from precomputed DNS fields
# Past-Frame Temporal Windowing Edition: (t-2, t-1, t) -> Q(t)
# ============================================================

import sys
from pathlib import Path

# Add repository root to sys.path for config import
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config

import time
import re
import gc
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import matplotlib.pyplot as plt

# 1. OPTIMIZE VRAM: Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 2. PREVENT VRAM HOGGING: Force memory growth 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth Enabled successfully.")
    except RuntimeError as e:
        print(f"Memory growth error: {e}")

# ============================================================
# 1. User settings & Safe Path Resolution
# ============================================================

NZ = getattr(config, 'NZ', 64)
NY = getattr(config, 'NY', 1024)
NT = getattr(config, 'NT', 324)

# 14 CHANNELS: 3 past/present frames of (u, v, theta, q) + 2 static parameter maps (Ra, Pr)
N_CHANNELS = 14  

DATASET_LABELS = [
    "Pr_2_Ra_1e4", "Pr_2_Ra_1e5", "Pr_2_Ra_1e6", "Pr_2_Ra_1e8", "Pr_2_Ra_1e9",
    "Pr_05_Ra_1e4", "Pr_05_Ra_1e5", "Pr_05_Ra_1e6", "Pr_05_Ra_1e8", "Pr_05_Ra_1e9",
    "Ra_1e4", "Ra_1e5", "Ra_1e6", "Ra_1e7", "Ra_1e8", "Ra_1e9"
]

DATASETS = {}
for label in DATASET_LABELS:
    # Check for direct file in processed directory or subfolder
    flat_path = config.PROCESSED_DATA_DIR / f"{label}_fields_Q.npz"
    subfolder_path = config.PROCESSED_DATA_DIR / label / f"{label}_fields_Q.npz"
    
    if subfolder_path.exists():
        DATASETS[label] = subfolder_path
    else:
        DATASETS[label] = flat_path

TEST_DATASET = "Ra_1e8"

BATCH_SIZE = 2
EPOCHS = 150
LEARNING_RATE = 1e-4

MODEL_OUT = config.MODEL_DIR / "model_out.keras"
NORM_OUT = config.MODEL_DIR / "normalization_constants.npz"


# ============================================================
# 2. Dataset class with Low-RAM Memory Footprint
# ============================================================

def extract_ra_pr(npz_data, label):
    if "Ra" in npz_data and "Pr" in npz_data:
        return float(npz_data["Ra"]), float(npz_data["Pr"])

    pr_match = re.search(r"Pr_(\d+)", label)
    if pr_match:
        pr_str = pr_match.group(1)
        pr_val = float(pr_str) if len(pr_str) == 1 else float(pr_str) / 10.0
    else:
        pr_val = 1.0

    ra_match = re.search(r"Ra_([0-9eE\+\-]+)", label)
    ra_val = float(ra_match.group(1)) if ra_match else 1e6

    return ra_val, pr_val


class RainyBenardQDataset:
    def __init__(self, dataset_dict, nz=64, ny=1024, nt=324):
        self.dataset_dict = dataset_dict
        self.nz = nz
        self.ny = ny
        self.nt = nt

        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None

    def load_one_npz(self, filename, label=""):
        filename_str = str(filename)
        data = np.load(filename_str)

        # Resolve corresponding _q.npz filename
        if "_fields_Q.npz" in filename_str:
            q_filename = filename_str.replace("_fields_Q.npz", "_q.npz")
        else:
            q_filename = filename_str.replace(".npz", "_q.npz")

        data_q = np.load(q_filename)

        u = data["u"]
        v = data["v"]
        theta = data["theta"]
        q = data_q["q"]
        Q = data["Q"]

        # Temporal Window: (t-2, t-1, t)
        u_tm2, u_tm1, u_t = np.moveaxis(u[:, :, :-2], -1, 0), np.moveaxis(u[:, :, 1:-1], -1, 0), np.moveaxis(u[:, :, 2:], -1, 0)
        v_tm2, v_tm1, v_t = np.moveaxis(v[:, :, :-2], -1, 0), np.moveaxis(v[:, :, 1:-1], -1, 0), np.moveaxis(v[:, :, 2:], -1, 0)
        th_tm2, th_tm1, th_t = np.moveaxis(theta[:, :, :-2], -1, 0), np.moveaxis(theta[:, :, 1:-1], -1, 0), np.moveaxis(theta[:, :, 2:], -1, 0)
        q_tm2, q_tm1, q_t = np.moveaxis(q[:, :, :-2], -1, 0), np.moveaxis(q[:, :, 1:-1], -1, 0), np.moveaxis(q[:, :, 2:], -1, 0)

        # Target Q at time t
        Q_target = np.moveaxis(Q[:, :, 2:], -1, 0)

        ra_val, pr_val = extract_ra_pr(data, label)
        ra_feature = np.log10(ra_val)
        pr_feature = float(pr_val)

        nt_valid = u_t.shape[0]
        ra_map = np.full((nt_valid, self.nz, self.ny), ra_feature, dtype=np.float32)
        pr_map = np.full((nt_valid, self.nz, self.ny), pr_feature, dtype=np.float32)

        X = np.stack([
            u_tm2, u_tm1, u_t,
            v_tm2, v_tm1, v_t,
            th_tm2, th_tm1, th_t,
            q_tm2, q_tm1, q_t,
            ra_map, pr_map
        ], axis=-1)

        Y = Q_target[..., np.newaxis]

        return X.astype(np.float32), Y.astype(np.float32)

    def split_by_dataset(self, test_label=TEST_DATASET, validation_fraction=0.2):
        if test_label not in self.dataset_dict:
            raise ValueError(f"Unknown TEST_DATASET '{test_label}'. Available: {list(self.dataset_dict.keys())}")

        X_train_list, Y_train_list = [], []
        X_val_list, Y_val_list = [], []
        X_test, Y_test = None, None

        for label, filename in self.dataset_dict.items():
            print(f"Loading {label}: {filename}")
            X_file, Y_file = self.load_one_npz(filename, label=label)

            if label == test_label:
                X_test, Y_test = X_file, Y_file
            else:
                n_total = X_file.shape[0]
                n_val = max(1, int(validation_fraction * n_total))
                n_train = n_total - n_val

                X_train_list.append(X_file[:n_train])
                Y_train_list.append(Y_file[:n_train])
                X_val_list.append(X_file[n_train:])
                Y_val_list.append(Y_file[n_train:])
                print(f"  -> Split: train={n_train}, val={n_val}")

            del X_file, Y_file
            gc.collect()

        print("\nConcatenating datasets into memory...")
        X_train = np.concatenate(X_train_list, axis=0)
        del X_train_list; gc.collect()

        Y_train = np.concatenate(Y_train_list, axis=0)
        del Y_train_list; gc.collect()

        X_val = np.concatenate(X_val_list, axis=0)
        del X_val_list; gc.collect()

        Y_val = np.concatenate(Y_val_list, axis=0)
        del Y_val_list; gc.collect()

        print("\nSplit completed:")
        print("Held-out test dataset:", test_label)
        print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
        print("X_val:  ", X_val.shape,   "Y_val:  ", Y_val.shape)
        print("X_test: ", X_test.shape,  "Y_test: ", Y_test.shape)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def fit_normalization(self, X_train, Y_train):
        self.X_mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
        self.X_std = X_train.std(axis=(0, 1, 2), keepdims=True)
        self.X_std = np.where(self.X_std == 0.0, 1.0, self.X_std)

        Y_interior = Y_train[:, 1:-1, :, :]
        self.Y_mean = Y_interior.mean(axis=(0, 1, 2), keepdims=True)
        self.Y_std = Y_interior.std(axis=(0, 1, 2), keepdims=True)
        self.Y_std = np.where(self.Y_std == 0.0, 1.0, self.Y_std)

        self.X_mean = self.X_mean.astype(np.float32)
        self.X_std = self.X_std.astype(np.float32)
        self.Y_mean = self.Y_mean.astype(np.float32)
        self.Y_std = self.Y_std.astype(np.float32)

    def normalize_X_inplace(self, X):
        np.subtract(X, self.X_mean, out=X)
        np.divide(X, self.X_std, out=X)

    def normalize_Y_inplace(self, Y):
        np.subtract(Y, self.Y_mean, out=Y)
        np.divide(Y, self.Y_std, out=Y)

    def save_normalization(self, filename=NORM_OUT):
        np.savez(filename, X_mean=self.X_mean, X_std=self.X_std, Y_mean=self.Y_mean, Y_std=self.Y_std)


# ============================================================
# 3. Data Generator
# ============================================================

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, Y, batch_size=BATCH_SIZE, shuffle=True):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.X[batch_indexes], self.Y[batch_indexes]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# ============================================================
# 4. U-Net Architecture
# ============================================================

def conv_block(x, n_filters, dropout=0.0):
    x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
    x = layers.Activation("relu")(x)
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)
    return x


def encoder_block(x, n_filters, dropout=0.0):
    skip = conv_block(x, n_filters, dropout=dropout)

    avg_pool = layers.AveragePooling2D(pool_size=(2, 2))(skip)
    max_pool = layers.MaxPooling2D(pool_size=(2, 2))(skip)

    pooled = layers.Add()([avg_pool, max_pool])

    return skip, pooled


def decoder_block(x, skip, n_filters, dropout=0.0):
    x = layers.Conv2DTranspose(n_filters, kernel_size=2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, n_filters, dropout=dropout)
    return x


def build_unet(input_shape=(NZ, NY, N_CHANNELS), base_filters=16):
    inputs = layers.Input(shape=input_shape)

    s1, p1 = encoder_block(inputs, base_filters)
    s2, p2 = encoder_block(p1, base_filters * 2)
    s3, p3 = encoder_block(p2, base_filters * 4)
    s4, p4 = encoder_block(p3, base_filters * 8, dropout=0.1)

    b1 = conv_block(p4, base_filters * 16, dropout=0.1)

    d1 = decoder_block(b1, s4, base_filters * 8, dropout=0.1)
    d2 = decoder_block(d1, s3, base_filters * 4)
    d3 = decoder_block(d2, s2, base_filters * 2)
    d4 = decoder_block(d3, s1, base_filters)

    outputs = layers.Conv2D(1, kernel_size=1, padding="same", activation="linear", name="Q_output")(d4)

    return models.Model(inputs, outputs, name="UNet_Q_regression_past")


# ============================================================
# 5. Masked losses and metrics
# ============================================================

def interior_mask_like(y_true):
    shape = tf.shape(y_true)
    batch, nz, ny, channels = shape[0], shape[1], shape[2], shape[3]
    z_ids = tf.range(nz)
    mask_z = tf.cast((z_ids > 0) & (z_ids < nz - 1), tf.float32)
    mask_z = tf.reshape(mask_z, (1, nz, 1, 1))
    return tf.tile(mask_z, (batch, 1, ny, channels))


def masked_mse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = interior_mask_like(y_true)
    sq = tf.square(y_true - y_pred) * mask
    return tf.reduce_sum(sq) / (tf.reduce_sum(mask) + 1e-8)


def masked_mae(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = interior_mask_like(y_true)
    abs_err = tf.abs(y_true - y_pred) * mask
    return tf.reduce_sum(abs_err) / (tf.reduce_sum(mask) + 1e-8)


def masked_gradient_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    dy_true = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    dy_pred = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    dz_true = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
    dz_pred = y_pred[:, :-1, :, :] - y_pred[:, :-1, :, :]
    loss_y = tf.reduce_mean(tf.square(dy_true - dy_pred))
    loss_z = tf.reduce_mean(tf.square(dz_true - dz_pred))
    return loss_y + loss_z


def combined_loss(y_true, y_pred):
    return masked_mse(y_true, y_pred) + 0.05 * masked_gradient_loss(y_true, y_pred)


def masked_correlation(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_i = y_true[:, 1:-1, :, :]
    y_pred_i = y_pred[:, 1:-1, :, :]
    y_true_flat = tf.reshape(y_true_i, [tf.shape(y_true_i)[0], -1])
    y_pred_flat = tf.reshape(y_pred_i, [tf.shape(y_pred_i)[0], -1])
    y_true_flat = y_true_flat - tf.reduce_mean(y_true_flat, axis=1, keepdims=True)
    y_pred_flat = y_pred_flat - tf.reduce_mean(y_pred_flat, axis=1, keepdims=True)
    numerator = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
    denominator = tf.sqrt(tf.reduce_sum(tf.square(y_true_flat), axis=1) * tf.reduce_sum(tf.square(y_pred_flat), axis=1))
    return tf.reduce_mean(numerator / (denominator + 1e-8))


# ============================================================
# 6. Channel ablation
# ============================================================

def channel_ablation_importance(model, X_test, Y_test, channel_names):
    test_gen = DataGenerator(X_test, Y_test, batch_size=BATCH_SIZE, shuffle=False)
    base_results = model.evaluate(test_gen, verbose=0)
    base_loss = base_results[0]

    print("\nBase test loss:", base_loss)
    print("\nChannel ablation importance:")

    importance = {}
    for c, name in enumerate(channel_names):
        X_ablate = X_test.copy()
        X_ablate[..., c] = 0.0

        ablate_gen = DataGenerator(X_ablate, Y_test, batch_size=BATCH_SIZE, shuffle=False)
        results = model.evaluate(ablate_gen, verbose=0)
        loss_c = results[0]
        delta = loss_c - base_loss
        importance[name] = {"loss": loss_c, "delta": delta}

        print(f"{name:15s}  loss = {loss_c:.4e}   delta = {delta:.4e}")
        del X_ablate, ablate_gen
        gc.collect()

    return base_loss, importance


# ============================================================
# 7. Main training script execution
# ============================================================

def main():
    train_start_time = time.time()

    dataset = RainyBenardQDataset(DATASETS, nz=NZ, ny=NY, nt=NT)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset.split_by_dataset()

    dataset.fit_normalization(X_train, Y_train)

    print("\nNormalizing datasets in-place...")
    dataset.normalize_X_inplace(X_train)
    dataset.normalize_X_inplace(X_val)
    dataset.normalize_X_inplace(X_test)

    dataset.normalize_Y_inplace(Y_train)
    dataset.normalize_Y_inplace(Y_val)
    dataset.normalize_Y_inplace(Y_test)

    dataset.save_normalization(NORM_OUT)

    model = build_unet(input_shape=(NZ, NY, N_CHANNELS), base_filters=16)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[masked_mse, masked_mae, masked_correlation]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_OUT), save_best_only=True, monitor="val_loss", mode="min"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1)
    ]

    print("\nCreating memory-efficient batch generators...")
    train_gen = DataGenerator(X_train, Y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_gen = DataGenerator(X_val, Y_val, batch_size=BATCH_SIZE, shuffle=False)

    print("\nStarting training loop...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # ============================================================
    # Plot & Save Training History
    # ============================================================
    print("\n--> Plotting Training vs Validation Loss...")
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_loss, label='Training Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2)
    
    plt.yscale('log')
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Combined Loss (Log Scale)', fontsize=11)
    plt.title('Model v013 (Past Window) Training & Validation Loss', fontsize=13, fontweight='bold')
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    loss_plot_file = config.OUTPUT_DIR / "training_loss.png"
    plt.savefig(loss_plot_file, dpi=300)
    plt.close()
    print(f"Saved loss curve plot to: {loss_plot_file}")

    # Save history dictionary safely to JSON
    history_json_file = config.OUTPUT_DIR / "history.json"
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}

    with open(history_json_file, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"Saved history data to: {history_json_file}")

    print("\nEvaluating on test set...")
    test_gen = DataGenerator(X_test, Y_test, batch_size=BATCH_SIZE, shuffle=False)
    test_results = model.evaluate(test_gen)

    print("\nTest results:")
    for name, value in zip(model.metrics_names, test_results):
        print(f"{name}: {value}")

    channel_names = [
        "u_tm2", "u_tm1", "u_t",
        "v_tm2", "v_tm1", "v_t",
        "theta_tm2", "theta_tm1", "theta_t",
        "q_tm2", "q_tm1", "q_t",
        "log10_Ra", "Pr"
    ]

    base_loss, ablation_results = channel_ablation_importance(model, X_test, Y_test, channel_names)

    summary_file = config.OUTPUT_DIR / "evaluation_summary.txt"
    with open(summary_file, "w") as f:
        f.write("========================================\n")
        f.write("          MODEL TEST METRICS            \n")
        f.write("========================================\n")
        for name, value in zip(model.metrics_names, test_results):
            f.write(f"{name:<25}: {value:.6e}\n")

        f.write("\n========================================\n")
        f.write("      CHANNEL ABLATION IMPORTANCE       \n")
        f.write("========================================\n")
        f.write(f"Base Test Loss           : {base_loss:.6e}\n\n")
        f.write(f"{'Channel':<20} {'Ablated Loss':<18} {'Delta (Importance)':<18}\n")
        f.write("-" * 58 + "\n")
        for name, res in ablation_results.items():
            f.write(f"{name:<20} {res['loss']:<18.6e} {res['delta']:<18.6e}\n")

    print(f"\nSaved evaluation metrics & ablation summary to: {summary_file}")
    print("\nDone.")
    print(f"Saved model: {MODEL_OUT}")
    print(f"Saved normalization: {NORM_OUT}")

    elapsed_time = time.time() - train_start_time
    print(f"\n Training Script Execution Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()
