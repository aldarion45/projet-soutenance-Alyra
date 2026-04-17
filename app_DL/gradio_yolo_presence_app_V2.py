from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import os
import socket
import subprocess
import torch
from ultralytics import YOLO

# -------------------------------------------------------------------
# Configuration générale de l'application
# -------------------------------------------------------------------
# Nom de la classe cible affiché dans l'interface.
# En français, "stair-climber" est le plus souvent appelé "stepper"
# (ou simulateur d'escalier).
TARGET_CLASS_NAME = "Stepper (simulateur d'escalier)"

# Dossier courant du script (sert à construire des chemins robustes).
BASE_DIR = Path(__file__).resolve().parent

# Modèles explicites, stockés dans le même dossier que ce script.
MODEL_1 = BASE_DIR / "modele_1.pt"
MODEL_2 = BASE_DIR / "modele_2.pt"


def pick_device() -> str:
    # Retourne "0" pour demander à Ultralytics d'utiliser le premier GPU CUDA.
    # Si CUDA n'est pas disponible, on force l'exécution sur CPU.
    # Utilise le GPU CUDA si disponible, sinon bascule sur CPU.
    return "0" if torch.cuda.is_available() else "cpu"


def get_compute_device_label(device: str) -> str:
    # Retourne un libellé humain pour l'affichage UI/logs.
    # Si on tourne sur CPU, pas besoin d'interroger nvidia-smi.
    if device == "cpu":
        return "CPU"

    # Tente d'abord via nvidia-smi (source la plus fiable en environnement Docker GPU).
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        gpu_names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if gpu_names:
            # DEVICE vaut "0" ici: on prend donc le premier GPU listé.
            return gpu_names[int(device)] if device.isdigit() and int(device) < len(gpu_names) else gpu_names[0]
    except Exception:
        # Silence volontaire: fallback PyTorch ci-dessous.
        pass

    # Fallback si nvidia-smi indisponible.
    try:
        return torch.cuda.get_device_name(int(device) if device.isdigit() else 0)
    except Exception:
        return f"GPU ({device})"


# Device global utilisé dans toutes les prédictions du script.
DEVICE = pick_device()
DEVICE_LABEL = get_compute_device_label(DEVICE)
# Cache mémoire des modèles YOLO déjà chargés.
# Objectif: accélérer les prédictions suivantes et éviter les rechargements disque.
_MODEL_CACHE = {}


def find_free_port(start: int = 7860, end: int = 7890) -> int:
    # Scanne une plage de ports locale (127.0.0.1) et retourne le premier disponible.
    # Utile pour éviter un crash si le port par défaut est déjà pris.
    # Cherche un port libre localement pour éviter les conflits au lancement.
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # connect_ex(...) retourne 0 si le port est occupé, sinon un code d'erreur.
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise OSError(f"No free port found in range {start}-{end}")


def _draw_box(
    image_bgr: np.ndarray,
    xyxy: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    # image_bgr: image au format OpenCV (BGR).
    # xyxy: coordonnées (x1, y1, x2, y2).
    # label: texte affiché sur la boîte (classe + confiance).
    # color: couleur de la boîte et du fond du texte.
    # Dessine la boîte englobante et un bandeau de texte lisible.
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)

    # Calcule la taille du texte pour dessiner un bandeau ajusté au-dessus de la boîte.
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    # Évite de sortir de l'image si la boîte touche le haut.
    y_top = max(0, y1 - th - 8)
    cv2.rectangle(image_bgr, (x1, y_top), (x1 + tw + 8, y1), color, -1)
    # Texte noir sur fond coloré pour un bon contraste visuel.
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


def _load_model(model_path: Path):
    # Fonction utilitaire centralisée pour:
    # 1) vérifier l'existence du modèle,
    # 2) le charger une seule fois,
    # 3) récupérer les noms de classes si disponibles.
    # Résout le chemin absolu du modèle et vérifie qu'il existe.
    resolved_path = model_path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Model not found: {resolved_path}")

    key = str(resolved_path)
    if key not in _MODEL_CACHE:
        # Mise en cache pour éviter de recharger le modèle à chaque inférence.
        _MODEL_CACHE[key] = YOLO(key)

    # Réutilise l'instance déjà chargée si elle est en cache.
    model = _MODEL_CACHE[key]
    # Certains modèles YOLO exposent `names` (mapping index -> nom de classe).
    names = getattr(model, "names", {}) or {}

    # Détermine un nom de classe "par défaut" pour les messages.
    # Si le mapping est vide, on retombe sur TARGET_CLASS_NAME.
    if isinstance(names, dict) and names:
        class_name = str(names.get(0, next(iter(names.values()))))
    else:
        class_name = TARGET_CLASS_NAME

    # Retourne tout ce qui est utile pour l'étape de prédiction.
    return model, names, class_name, resolved_path


def _predict_single_model(
    image: np.ndarray,
    conf_threshold: float,
    model_path: Path,
    model_label: str,
):
    # Entrées:
    # - image: image RGB venant de Gradio.
    # - conf_threshold: seuil de confiance appliqué par YOLO.
    # - model_path: chemin du .pt à utiliser.
    # - model_label: identifiant lisible dans les sorties texte.
    #
    # Sorties:
    # - image annotée (RGB),
    # - texte détaillé de détection.
    # Lance une prédiction YOLO avec un seul modèle puis prépare les sorties.
    model, model_class_names, class_name, resolved_path = _load_model(model_path)
    # Ultralytics retourne une liste de résultats; ici on traite la première image.
    result = model.predict(source=image, conf=conf_threshold, device=DEVICE, verbose=False)[0]

    # OpenCV dessine en BGR; conversion nécessaire avant annotation.
    annotated_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if len(result.boxes) == 0:
        # Cas sans détection: on renvoie l'image d'origine + un statut détaillé.
        status = (
            f"[{model_label}] NOT DETECTED | class={class_name} | confidence=0.00 | "
            f"threshold={conf_threshold:.2f} | device={DEVICE_LABEL}\n"
            f"model={resolved_path}"
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return annotated_rgb, status

    confs = result.boxes.conf.detach().cpu().numpy().tolist()
    xyxy_list = result.boxes.xyxy.detach().cpu().numpy().tolist()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
    # On copie explicitement les tenseurs sur CPU puis en NumPy pour simplifier
    # les traitements Python (tri, formatage, affichage).

    # Détection la plus confiante pour le résumé principal.
    best_idx = int(np.argmax(confs))
    best_conf = float(confs[best_idx])

    # Liste structurée des détections (utilisée ensuite pour trier et afficher).
    detections = []

    for box, conf, cls_idx in zip(xyxy_list, confs, classes):
        # Conversion en entiers pour tracer proprement la bbox sur l'image.
        x1, y1, x2, y2 = [int(v) for v in box]
        # Récupère le nom de la classe si connu, sinon fallback sur class_name.
        cls_name = str(model_class_names.get(cls_idx, class_name))
        label = f"{cls_name}: {conf:.2f}"
        _draw_box(annotated_bgr, (x1, y1, x2, y2), label)
        detections.append(
            {
                "class_name": cls_name,
                "confidence": float(conf),
                "xyxy": (x1, y1, x2, y2),
            }
        )

    # Trie décroissant par confiance pour présenter d'abord les meilleures détections.
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # Construction d'un rapport lisible, trié de la meilleure à la moins bonne détection.
    detection_lines = []
    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["xyxy"]
        detection_lines.append(
            f"{i}. class={det['class_name']} | conf={det['confidence']:.2f} | bbox=({x1},{y1},{x2},{y2})"
        )

    status = (
        # Ligne de synthèse globale.
        f"[{model_label}] DETECTED | best_confidence={best_conf:.2f} | boxes={len(xyxy_list)} | "
        f"threshold={conf_threshold:.2f} | device={DEVICE_LABEL}\n"
        # Chemin du modèle pour faciliter le debug/reproductibilité.
        f"model={resolved_path}\n"
        # Liste détaillée de toutes les boîtes détectées.
        + "\n".join(detection_lines)
    )

    # Retour au format RGB attendu par Gradio pour l'affichage.
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, status


def predict_both_models(image: Optional[np.ndarray], conf_threshold: float):
    # Callback principal appelé par le bouton Gradio.
    # Cette fonction compare les 2 modèles sur exactement la même image et
    # renvoie 4 sorties: image+texte pour modèle_1, image+texte pour modèle_2.
    # Point d'entrée Gradio: exécute les 2 modèles et retourne les 4 sorties UI.
    if image is None:
        # Message utilisateur explicite si aucun fichier n'est fourni.
        msg = "Veuillez d'abord importer une image."
        return None, msg, None, msg

    try:
        # Exécute le modèle 1.
        out_model_1_img, out_model_1_txt = _predict_single_model(
            image=image,
            conf_threshold=conf_threshold,
            model_path=MODEL_1,
            model_label="modele_1",
        )
        # Exécute le modèle 2 sur la même image pour comparaison.
        out_model_2_img, out_model_2_txt = _predict_single_model(
            image=image,
            conf_threshold=conf_threshold,
            model_path=MODEL_2,
            model_label="modele_2",
        )
    except Exception as exc:
        # Gestion d'erreur globale: on évite de faire planter l'interface.
        # Une erreur de chargement/prédiction est affichée des deux côtés de l'UI.
        err = f"Erreur de chargement/prédiction du modèle : {exc}"
        return None, err, None, err

    # Retour dans l'ordre exact attendu par `outputs=[...]` plus bas.
    return out_model_1_img, out_model_1_txt, out_model_2_img, out_model_2_txt


# -------------------------------------------------------------------
# Construction de l'interface Gradio
# -------------------------------------------------------------------
with gr.Blocks(title=f"Détecteur {TARGET_CLASS_NAME} (YOLO) - V2") as demo:
    # Interface Gradio: entrée image + seuil, puis comparaison visuelle/texte des 2 modèles.
    gr.Markdown(f"# {TARGET_CLASS_NAME} - Comparaison (V2)")
    gr.Markdown(
        "Comparaison automatique de 2 modèles sur la même image : `modele_1` vs `modele_2`.  \n"
        "`modele_1`est le premier modèle entraîné.  \n"
        "`modele_2`est le modèle réentrainé avec les faux positifs  \n"
        "Un *stepper* (aussi appelé *stair-climber*) est un appareil de cardio qui simule "
        "la montée d'escaliers.  \n"
        f"**Appareil de calcul :** `{DEVICE_LABEL}`"
    )

    # Zone d'import de l'image (convertie en tableau NumPy RGB par Gradio).
    input_image = gr.Image(type="numpy", label="Image d'entrée")
    # Curseur de seuil de confiance appliqué à l'inférence YOLO.
    conf_slider = gr.Slider(
        minimum=0.05,
        maximum=0.95,
        value=0.75,
        step=0.01,
        label="Seuil de confiance",
    )
    # Déclenche la comparaison.
    run_btn = gr.Button("Lancer la comparaison", variant="primary")

    # Présentation en deux colonnes pour comparer les modèles côte à côte.
    with gr.Row():
        with gr.Column():
            # Colonne gauche: sorties du modèle_1.
            output_freeze_image = gr.Image(type="numpy", label="Prédiction - modele_1")
            output_freeze_text = gr.Textbox(label="Résultat - modele_1", lines=14)

        with gr.Column():
            # Colonne droite: sorties du modèle_2.
            output_traps_image = gr.Image(type="numpy", label="Prédiction - modele_2")
            output_traps_text = gr.Textbox(label="Résultat - modele_2", lines=14)

    # Branche le bouton au callback + mappe les entrées/sorties.
    run_btn.click(
        fn=predict_both_models,
        inputs=[input_image, conf_slider],
        outputs=[
            output_freeze_image,
            output_freeze_text,
            output_traps_image,
            output_traps_text,
        ],
    )


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Point d'entrée d'exécution locale / Docker
    # ----------------------------------------------------------------
    # Permet de fixer le port via variable d'environnement, sinon recherche auto.
    env_port = os.getenv("GRADIO_SERVER_PORT")
    server_port = int(env_port) if env_port else find_free_port(7860, 7890)
    # Log minimal pour savoir quel port est réellement utilisé.
    print(f"Launching Gradio on port {server_port}")
    # server_name="0.0.0.0" rend l'app accessible depuis l'extérieur du conteneur.
    demo.launch(server_name="0.0.0.0", server_port=server_port)
