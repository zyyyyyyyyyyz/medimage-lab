"""Architecture U-Net extraite du notebook TP4 exécuté (BUSI, 256x256, 1 canal).
Le corps de simple_unet_model est conservé tel quel depuis le TP.
"""
try:
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate
    from tensorflow.keras.models import Model
except ImportError:  # Preprocessing may be tested even when TensorFlow is optional.
    Input = Conv2D = MaxPooling2D = Conv2DTranspose = Dropout = concatenate = Model = None

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # ---------------- ENCODER (contracting path) ----------------
    # Block 1 : 16 filters
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    # Block 2 : 32 filters
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Block 3 : 64 filters
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Block 4 : 128 filters
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Block 5 : 256 filters (bottleneck, NO pooling)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # ---------------- DECODER (expanding path) ----------------
    # Up 1 : 128 filters, skip connection with c4
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    # Up 2 : 64 filters, skip connection with c3
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    # Up 3 : 32 filters, skip connection with c2
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    # Up 4 : 16 filters, skip connection with c1
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    # ---------------- OUTPUT ----------------
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def preprocess_busi_like_tp(image_gray):
    """Reproduit normalize(image_dataset, axis=1) sur une image batchée.

    TP d'origine : image_dataset.shape=(N, H, W), normalize(axis=1),
    puis np.expand_dims(..., axis=3). L'axe 1 est donc la HAUTEUR.
    Ne pas remplacer par un simple /255, qui change l'entrée du modèle.
    """
    import cv2
    import numpy as np
    # TP4 cell 18 uses cv2.resize(img, (SIZE, SIZE)) => OpenCV default INTER_LINEAR.
    resized = cv2.resize(image_gray.astype(np.uint8), (256, 256), interpolation=cv2.INTER_LINEAR)
    batch = resized[np.newaxis, :, :].astype(np.float32)
    try:
        # Exact TP4 call when the optional TensorFlow inference environment exists.
        from tensorflow.keras.utils import normalize
        batch = normalize(batch, axis=1)
    except ImportError:
        # Numerically equivalent L2 fallback keeps conventional tests TensorFlow-free.
        norms = np.linalg.norm(batch, ord=2, axis=1, keepdims=True)
        batch = np.divide(batch, np.maximum(norms, 1e-12))
    return batch[:, :, :, np.newaxis]


def load_busi_model(model_path):
    """Charge uniquement un chemin local administré; compile=False évite l'optimizer historique."""
    from tensorflow.keras.models import load_model
    model = load_model(str(model_path), compile=False)
    if tuple(model.input_shape[1:]) != (256, 256, 1):
        raise ValueError(f"Entrée inattendue : {model.input_shape}; attendu (None,256,256,1)")
    if tuple(model.output_shape[1:]) != (256, 256, 1):
        raise ValueError(f"Sortie inattendue : {model.output_shape}; attendu (None,256,256,1)")
    return model


def predict_busi_outputs(model, original_gray):
    """Retourne la probabilité et le masque TP4 aux dimensions originales.

    Le seuil est volontairement fixe et strict (``prediction > 0.5``), comme
    dans les cellules d'inférence du TP4. La carte de probabilité est
    redimensionnée uniquement pour l'affichage; le seuillage est effectué
    sur la sortie native 256×256 du réseau.
    """
    import cv2
    import numpy as np
    batch = preprocess_busi_like_tp(original_gray)
    pred = np.asarray(model.predict(batch, verbose=0))[0, :, :, 0]
    if pred.shape != (256, 256) or not np.isfinite(pred).all():
        raise ValueError(f"Prédiction U-Net invalide : forme {pred.shape}")
    # TP4 cells 33/38 use a strict prediction > 0.5 threshold.
    binary = (pred > 0.5).astype(np.uint8) * 255
    output_size = (original_gray.shape[1], original_gray.shape[0])
    probability = cv2.resize(
        pred.astype(np.float32), output_size, interpolation=cv2.INTER_LINEAR
    )
    mask = cv2.resize(binary, output_size, interpolation=cv2.INTER_NEAREST)
    return probability, mask


def predict_busi(model, original_gray, threshold=0.5):
    """Compatibilité API: retourne le masque TP4 avec le seuil exact 0.5."""
    if float(threshold) != 0.5:
        raise ValueError("Le seuil du TP4 est fixe : prediction > 0.5")
    _, mask = predict_busi_outputs(model, original_gray)
    return mask
