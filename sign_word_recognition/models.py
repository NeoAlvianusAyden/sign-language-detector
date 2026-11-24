"""Hanya dua arsitektur: CNN (dari nol) dan MobileNetV2 (transfer learning ImageNet).
Augmentasi & rescaling ada DI DALAM model, jadi training, evaluasi, dan webcam memakai
preprocessing yang identik. Tidak ada flip horizontal (menjaga makna tangan kiri/kanan)."""
from tensorflow import keras
from tensorflow.keras import layers

BANNED = ("LSTM", "GRU", "RNN", "Bidirectional", "Attention", "Transformer")


def assert_no_recurrent(model):
    def walk(m):
        for l in getattr(m, "layers", []):
            name = type(l).__name__
            if any(b.lower() in name.lower() for b in BANNED):
                raise RuntimeError(f"Layer terlarang terdeteksi: {name}")
            walk(l)
    walk(model)


def augmentation():
    return keras.Sequential([
        layers.RandomRotation(0.04),
        layers.RandomTranslation(0.06, 0.06),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.15, value_range=(0, 255)),
    ], name="augmentation")


def build_cnn(size, num_classes, filters=(32, 64, 128, 128), dense=128, dropout=0.4):
    """Baseline CNN kecil: Conv-BN-ReLU-MaxPool-Dropout x4 -> GAP -> Dense -> Softmax."""
    inp = layers.Input((size, size, 3))
    x = augmentation()(inp)
    x = layers.Rescaling(1 / 255.0)(x)
    for f in filters:
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.15)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    m = keras.Model(inp, out, name="cnn_baseline")
    assert_no_recurrent(m)
    return m


def build_mobilenetv2(size, num_classes, dropout=0.3):
    """MobileNetV2 pretrained ImageNet. Rescaling(1/127.5, -1) == mobilenet_v2.preprocess_input."""
    base = keras.applications.MobileNetV2(input_shape=(size, size, 3), include_top=False,
                                          weights="imagenet")
    base.trainable = False
    inp = layers.Input((size, size, 3))
    x = augmentation()(inp)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    m = keras.Model(inp, out, name="mobilenetv2_transfer")
    assert_no_recurrent(m)
    return m, base


def set_trainable_top(base, n_layers):
    """Fine-tuning bertahap: buka n_layers teratas; BatchNorm tetap beku."""
    base.trainable = True
    for l in base.layers[:-n_layers]:
        l.trainable = False
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
