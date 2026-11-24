import json
import pandas as pd
import tensorflow as tf
from config import OUT, ROOT, SEED


def load_classes():
    return json.loads((OUT / "classes.json").read_text())


def load_frames_df(split=None):
    df = pd.read_csv(OUT / "frames.csv")
    if split:
        df = df[df.split == split]
    return df.reset_index(drop=True)


def make_ds(df, size, batch=32, shuffle=False):
    paths = [str(ROOT / p) for p in df.frame_path]
    labels = df.label_idx.values.astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)

    def _load(p, y):
        img = tf.io.decode_jpeg(tf.io.read_file(p), channels=3)
        img = tf.image.resize(img, (size, size), antialias=True)
        return tf.cast(img, tf.float32), y

    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)
