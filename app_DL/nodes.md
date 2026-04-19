# Notes de soutenance: API Flask YOLO vs Gradio

## 1) Fonctionnement API du script `demo_api_ui_yolo_flask.py`

### Vue d'ensemble
Le script implémente une mini application web avec Flask.
Il expose deux endp0onbh8579œ&  ;÷ø»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»ÊÊ³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³¥ ?ºººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººº¿_=oè(i=-
- `GET /` : affiche une page HTML avec formulaire d'upload image + seuil de confiance.
- `POST /predict` : reçoit l'image, lance la prédiction YOLO, renvoie la même page enrichie avec résultats.

### Cycle de vie au démarrage
1. Le script résout le chemin du modèle (`YOLO_MODEL_PATH` ou `modele_2.pt`).
2. Il choisit le device (`cuda:0` si GPU dispo, sinon `cpu`).
3. Il charge le modèle YOLO une seule fois.
4. Il démarre le serveur Flask sur `0.0.0.0:8000`.

### Flux détaillé d'une requête de prédiction
1. Le navigateur envoie un formulaire `multipart/form-data` sur `POST /predict`.
2. Le backend lit:
- le fichier image (`request.files['image']`)
- le seuil de confiance (`request.form['conf']`)
3. Le backend valide/nettoie les données:
- cast du seuil en float
- clamp du seuil dans `[0.05, 0.95]`
- décodage image binaire -> `numpy` -> matrice OpenCV BGR
4. Le backend appelle `run_prediction(...)`:
- conversion BGR -> RGB
- `model.predict(...)`
- extraction des boîtes/classes/confiances
- dessin des bounding boxes sur l'image
- tri des détections par confiance
5. Le backend convertit l'image annotée en JPEG base64.
6. Le backend retourne une page HTML serveur contenant:
- message de statut
- image annotée
- tableau des détections

### Contrat de sortie (dans cette démo)
Cette démo n'expose pas un JSON API public.
Elle renvoie du HTML rendu serveur avec les résultats.

## 2) Comment serait une vraie API de production

### API REST typique
- `POST /api/predict`
- Entrée: image + seuil
- Sortie JSON:
  - nom modèle
  - device
  - liste d'objets détectés
  - bboxes

Exemple de sortie:
```json
{
  "model": "modele_2.pt",
  "device": "cuda:0",
  "detections": [
    {"class": "stepper", "confidence": 0.91, "bbox": [120, 80, 330, 420]}
  ]
}
```

Optionnel:
- endpoint `GET /api/health` pour supervision
- endpoint `POST /api/predict/image` qui renvoie directement une image annotée

## 3) Fonctionnement global de Gradio

### Principe
Gradio relie des composants UI (image, slider, bouton, textbox) à une fonction Python callback.

Dans ton application:
1. L'utilisateur charge une image et règle le seuil.
2. Le clic sur `run_btn` déclenche `predict_both_models(...)`.
3. Cette fonction appelle tes deux modèles successivement.
4. Gradio injecte automatiquement les retours dans les composants de sortie.

### Caractéristiques importantes
- Très rapide à prototyper: pas besoin de coder tout le front HTML/CSS/JS.
- Mapping entrées/sorties explicite via `inputs=[...]`, `outputs=[...]`.
- Serveur web intégré prêt à l'emploi (`demo.launch(...)`).
- Endpoints internes gérés par Gradio (mais pas API produit métier documentée par défaut).

## 4) Pourquoi avoir préféré Gradio pour la soutenance

### Argumentaire technique simple
1. **Vitesse de mise en œuvre**
Tu as pu démontrer le modèle rapidement sans développer un front séparé.

2. **Lisibilité pédagogique**
Le lien "input utilisateur -> fonction Python -> output visuel" est direct et facile à expliquer au jury.

3. **Comparaison de modèles facilitée**
Ton interface affiche `modele_1` vs `modele_2` côte à côte avec images annotées et texte.

4. **Réduction du risque de démo**
Moins de couches techniques (frontend, API gateway, docs OpenAPI, auth) donc moins de points de panne.

5. **Cohérence avec l'objectif**
Pour une soutenance orientée IA, Gradio met l'accent sur la valeur modèle et les résultats, pas sur l'infrastructure.

### Limites
- Gradio n'est pas une API métier de production à lui seul.
- Pour industrialiser: FastAPI/Flask API dédiée + contrat JSON + sécurité + monitoring + versioning.

## 5) Conclusion

"J'ai choisi Gradio pour accélérer la démonstration de l'inférence YOLO: l'interface, le callback Python et l'affichage des bounding boxes sont intégrés dans un même script. C'était le meilleur compromis pour une soutenance: livrer vite, visualiser clairement les résultats, et comparer deux modèles. En production, je séparerais ensuite en API REST dédiée avec sorties JSON, sécurité et supervision." 
