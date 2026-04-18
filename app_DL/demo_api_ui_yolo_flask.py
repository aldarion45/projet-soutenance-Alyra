from __future__ import annotations

# Encodage base64: utilisé pour intégrer l'image annotée directement dans le HTML
# sans écrire de fichier temporaire sur disque.
import base64
import os
from pathlib import Path
from typing import Any

# OpenCV: décodage image, conversions couleur et dessin des bounding boxes.
import cv2
# NumPy: manipulation de buffer binaire et tableaux image.
import numpy as np
# Torch: détection GPU CUDA disponible ou fallback CPU.
import torch
# Flask: micro-framework web pour exposer une UI simple et un endpoint de prédiction.
from flask import Flask, render_template_string, request
# Ultralytics YOLO: moteur d'inférence objet.
from ultralytics import YOLO

# Dossier du script courant (référence stable pour retrouver le modèle localement).
BASE_DIR = Path(__file__).resolve().parent
# Chemin modèle:
# - prioritairement via variable d'environnement YOLO_MODEL_PATH (utile en Docker/CI),
# - sinon fallback sur modele_2.pt dans le même dossier.
MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", str(BASE_DIR / "modele_2.pt"))).expanduser().resolve()
# Sélection du device d'inférence:
# - "cuda:0" si un GPU compatible est disponible,
# - sinon exécution CPU.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# Nom cible par défaut si le mapping de classes du modèle ne contient pas la classe.
TARGET_CLASS_NAME = "Stepper (simulateur d'escalier)"

# Vérification précoce: on échoue au démarrage si le fichier modèle est absent
# pour éviter un crash plus tard au premier appel utilisateur.
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Chargement du modèle une seule fois au démarrage du serveur.
# Avantage: évite un coût de chargement important à chaque requête HTTP.
model = YOLO(str(MODEL_PATH))
# Récupération des noms de classes exposés par le modèle (si disponibles).
model_names = getattr(model, "names", {}) or {}

# Initialisation de l'application Flask.
app = Flask(__name__)

# Template HTML embarqué (inline) pour garder une démo autonome dans un seul fichier.
# Le rendu se fait avec Jinja2 via render_template_string.
HTML_PAGE = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>YOLO Demo API + UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }
    .card { background: white; border-radius: 12px; padding: 16px; max-width: 960px; box-shadow: 0 8px 24px rgba(15,23,42,.08); }
    h1 { margin-top: 0; }
    .meta { color: #475569; font-size: 14px; margin-bottom: 16px; }
    form { display: flex; gap: 12px; flex-wrap: wrap; align-items: end; margin-bottom: 16px; }
    label { display: block; font-size: 14px; color: #334155; margin-bottom: 6px; }
    input[type=file], input[type=number] { padding: 8px; }
    button { padding: 10px 14px; border: 0; border-radius: 8px; background: #0f766e; color: #fff; cursor: pointer; }
    button:hover { background: #115e59; }
    .error { color: #b91c1c; margin: 8px 0; }
    .ok { color: #065f46; margin: 8px 0; }
    img { max-width: 100%; border-radius: 10px; border: 1px solid #e2e8f0; }
    table { border-collapse: collapse; width: 100%; margin-top: 14px; background: #fff; }
    th, td { border: 1px solid #e2e8f0; padding: 8px; text-align: left; font-size: 14px; }
    th { background: #f1f5f9; }
    code { background: #e2e8f0; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Démo YOLO: upload image et bounding boxes</h1>
    <div class="meta">
      Modèle: <code>{{ model_path }}</code> | Device: <code>{{ device }}</code>
    </div>

    <form method="post" action="/predict" enctype="multipart/form-data">
      <div>
        <label for="image">Image</label>
        <input id="image" name="image" type="file" accept="image/*" required />
      </div>
      <div>
        <label for="conf">Seuil de confiance (0.05 à 0.95)</label>
        <input id="conf" name="conf" type="number" min="0.05" max="0.95" step="0.01" value="0.50" />
      </div>
      <button type="submit">Prédire</button>
    </form>

    {% if error %}
      <p class="error">{{ error }}</p>
    {% endif %}

    {% if status %}
      <p class="ok">{{ status }}</p>
    {% endif %}

    {% if image_b64 %}
      <h3>Image annotée</h3>
      <img src="data:image/jpeg;base64,{{ image_b64 }}" alt="Image annotée" />
    {% endif %}

    {% if detections %}
      <h3>Détections</h3>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Classe</th>
            <th>Confiance</th>
            <th>Bounding box (x1, y1, x2, y2)</th>
          </tr>
        </thead>
        <tbody>
          {% for d in detections %}
          <tr>
            <td>{{ loop.index }}</td>
            <td>{{ d.class_name }}</td>
            <td>{{ "%.3f"|format(d.confidence) }}</td>
            <td>({{ d.bbox[0] }}, {{ d.bbox[1] }}, {{ d.bbox[2] }}, {{ d.bbox[3] }})</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    {% endif %}
  </div>
</body>
</html>
"""


def draw_box(image_bgr: np.ndarray, bbox: tuple[int, int, int, int], label: str) -> None:
    # Fonction utilitaire de visualisation:
    # - dessine le rectangle de détection,
    # - ajoute un cartouche texte (classe + confiance).
    # Note: OpenCV attend des images en BGR.
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0)
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)

    # Calcul des dimensions du texte pour dimensionner le fond.
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    # Protection bord supérieur: évite de dessiner hors image.
    y_top = max(0, y1 - th - 8)
    cv2.rectangle(image_bgr, (x1, y_top), (x1 + tw + 8, y1), color, -1)
    cv2.putText(
        image_bgr,
        label,
        (x1 + 4, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def run_prediction(image_bgr: np.ndarray, conf_threshold: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    # Pipeline d'inférence principal.
    # Entrée: image BGR décodée depuis la requête HTTP.
    # 1) Conversion en RGB (format attendu par Ultralytics pour source numpy).
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # 2) Appel modèle YOLO.
    # - conf: seuil de confiance appliqué côté modèle.
    # - device: GPU/CPU choisi au démarrage.
    # - verbose=False: sortie console minimale.
    result = model.predict(source=image_rgb, conf=conf_threshold, device=DEVICE, verbose=False)[0]

    # Structure de sortie normalisée pour l'affichage HTML.
    detections: list[dict[str, Any]] = []
    if len(result.boxes) == 0:
        # Aucun objet détecté: on renvoie l'image telle quelle + liste vide.
        return image_bgr, detections

    # Extraction des tenseurs YOLO vers Python natif:
    # - confs: score de confiance,
    # - xyxy: coordonnées (x1,y1,x2,y2),
    # - classes: index de classe.
    confs = result.boxes.conf.detach().cpu().numpy().tolist()
    xyxy_list = result.boxes.xyxy.detach().cpu().numpy().tolist()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()

    for box, conf, cls_idx in zip(xyxy_list, confs, classes):
        # Cast en int pour un tracé propre et un affichage lisible.
        x1, y1, x2, y2 = [int(v) for v in box]
        # Mapping index -> nom de classe (fallback sur nom cible si absent).
        class_name = str(model_names.get(cls_idx, TARGET_CLASS_NAME))
        # Ajout d'une ligne de résultat structurée.
        detections.append(
            {
                "class_name": class_name,
                "confidence": float(conf),
                "bbox": (x1, y1, x2, y2),
            }
        )
        # Annotation visuelle directe sur l'image.
        draw_box(image_bgr, (x1, y1, x2, y2), f"{class_name}: {conf:.2f}")

    # Tri décroissant pour présenter d'abord les prédictions les plus fiables.
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return image_bgr, detections


def to_b64_jpeg(image_bgr: np.ndarray) -> str:
    # Convertit l'image annotée en JPEG puis en base64.
    # Cela permet d'inclure l'image dans le HTML via une data URL.
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("Unable to encode image to JPEG")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


@app.get("/")
def index():
    # Endpoint d'accueil:
    # - affiche le formulaire d'upload,
    # - ne montre aucune prédiction par défaut.
    return render_template_string(
        HTML_PAGE,
        model_path=str(MODEL_PATH),
        device=DEVICE,
        image_b64=None,
        detections=None,
        status=None,
        error=None,
    )


@app.post("/predict")
def predict():
    # Endpoint de traitement:
    # reçoit l'image et le seuil, exécute l'inférence, renvoie la page complétée.

    # Récupération du seuil dans le formulaire (string -> float).
    conf_raw = request.form.get("conf", "0.50")
    try:
        conf_threshold = float(conf_raw)
    except ValueError:
        # Sécurité: fallback si saisie non numérique.
        conf_threshold = 0.50

    # Clamp de sécurité pour rester dans une plage raisonnable.
    conf_threshold = max(0.05, min(0.95, conf_threshold))

    # Récupération du fichier image envoyé en multipart/form-data.
    uploaded = request.files.get("image")
    if not uploaded:
        # Réponse UI explicite si aucune image n'est transmise.
        return render_template_string(
            HTML_PAGE,
            model_path=str(MODEL_PATH),
            device=DEVICE,
            image_b64=None,
            detections=None,
            status=None,
            error="Aucune image reçue.",
        )

    # Lecture binaire brute puis conversion en tableau NumPy uint8.
    raw = uploaded.read()
    np_buf = np.frombuffer(raw, dtype=np.uint8)
    # Décodage du buffer image vers matrice OpenCV BGR.
    image_bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)

    if image_bgr is None:
        # Cas de format invalide ou fichier corrompu.
        return render_template_string(
            HTML_PAGE,
            model_path=str(MODEL_PATH),
            device=DEVICE,
            image_b64=None,
            detections=None,
            status=None,
            error="Format d'image non supporté.",
        )

    # Exécution de l'inférence et récupération de l'image annotée + métadonnées.
    annotated, detections = run_prediction(image_bgr, conf_threshold)
    # Encodage pour affichage direct dans la balise <img>.
    image_b64 = to_b64_jpeg(annotated)

    # Message de synthèse affiché en haut des résultats.
    if detections:
        status = f"{len(detections)} détection(s) trouvée(s) avec un seuil={conf_threshold:.2f}."
    else:
        status = f"Aucune détection avec un seuil={conf_threshold:.2f}."

    return render_template_string(
        HTML_PAGE,
        model_path=str(MODEL_PATH),
        device=DEVICE,
        image_b64=image_b64,
        detections=detections,
        status=status,
        error=None,
    )


if __name__ == "__main__":
    # Point d'entrée local:
    # - host 0.0.0.0: accessible depuis l'extérieur du conteneur/VM.
    # - port 8000: port HTTP de la démo.
    # - debug=False: mode production simple (pas de reloader/verbeux dev).
    #
    # Interface disponible via:
    # http://127.0.0.1:8000
    app.run(host="0.0.0.0", port=8000, debug=False)
