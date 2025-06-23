import os
import uuid
import json
import requests # <-- AÑADIDO
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import texttospeech
from google.cloud import storage
import vertexai # <-- AÑADIDO
from vertexai.preview.vision_models import ImageGenerationModel # <-- AÑADIDO

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv()

# Lógica para manejar credenciales de Google Cloud en producción
if os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON'):
    credentials_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    credentials_path = '/tmp/gcp-credentials.json'
    with open(credentials_path, 'w') as f:
        f.write(credentials_json_str)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

app = Flask(__name__)
CORS(app)

# Configuración de APIs de Google
try:
    # Configuración de Gemini y Clientes de Cloud
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    text_to_speech_client = texttospeech.TextToSpeechClient()
    storage_client = storage.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

    # --- NUEVO: Inicialización de Vertex AI ---
    # Se usa para la generación de imágenes con el modelo Imagen 2.
    # Requiere el ID del proyecto y una región (ej. 'us-central1').
    vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv("GCP_REGION", "us-central1"))

except Exception as e:
    print(f"Error al configurar los clientes de Google: {e}")

# Modelos de IA
model_text = genai.GenerativeModel('gemini-1.5-flash')
# --- NUEVO: Se carga el modelo de generación de imágenes ---
model_image = ImageGenerationModel.from_pretrained("imagegeneration@006")


# --- 2. FUNCIONES AUXILIARES ---

def upload_to_gcs(file_stream, destination_blob_name, content_type='audio/mpeg'):
    """Sube un stream de archivo a un bucket de GCS y lo hace público."""
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        # Sube el contenido desde el string de bytes
        blob.upload_from_string(file_stream, content_type=content_type)
        blob.make_public()
        return blob.public_url
    except Exception as e:
        print(f"Error al subir a GCS: {e}")
        return None

def safe_json_parse(text):
    """Intenta parsear un string JSON, limpiando el formato si es necesario."""
    text = text.strip().replace('```json', '').replace('```', '')
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Error al decodificar JSON: {e}")
        print(f"Texto problemático: {text}")
        return None

# --- 3. ENDPOINTS DE LA API ---

@app.route("/")
def index():
    return "Backend de IA para Videos - ¡Corriendo!"

@app.route('/api/generate-initial-content', methods=['POST'])
def generate_initial_content():
    data = request.get_json()
    prompt = f"""
    Eres un experto guionista de videos para redes sociales.
    Crea un guion para un video sobre el tema: "{data.get('guionPersonalizado', 'Un día en la vida')}".
    El video debe durar aproximadamente {data.get('duracionVideo', 30)} segundos.
    El tono debe ser para el nicho de '{data.get('nicho', 'Tecnología')}'.
    El idioma del guion debe ser '{data.get('idioma', 'es')}'
    Genera una estructura JSON que contenga una única clave "scenes".
    El valor de "scenes" debe ser un array de objetos. Cada objeto representa una escena y debe tener:
    1. Una clave "id" con un UUID único (ej: "scene-1", "scene-2").
    2. Una clave "script" con el texto de la narración para esa escena. El guion debe ser conciso y directo.
    Asegúrate de que la salida sea únicamente el objeto JSON válido y nada más.
    """
    try:
        response = model_text.generate_content(prompt)
        parsed_json = safe_json_parse(response.text)
        if parsed_json and 'scenes' in parsed_json:
            for i, scene in enumerate(parsed_json['scenes']):
                scene['id'] = scene.get('id', f'scene-{uuid.uuid4()}')
            return jsonify(parsed_json)
        else:
            return jsonify({"error": "La respuesta del modelo no tuvo el formato esperado."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/regenerate-scene-part', methods=['POST'])
def regenerate_scene_part():
    data = request.get_json()
    scene = data.get('scene')
    part = data.get('part')
    if not scene or not part:
        return jsonify({"error": "Faltan datos de escena o parte a regenerar"}), 400

    if part == 'script':
        prompt = f"Eres un guionista. Reescribe el siguiente guion para una escena de video de forma creativa, manteniendo la idea central. Guion original: '{scene.get('script')}'. Devuelve solo el texto del nuevo guion, sin comillas ni texto introductorio."
        try:
            response = model_text.generate_content(prompt)
            return jsonify({"newScript": response.text.strip()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    # --- SECCIÓN COMPLETAMENTE ACTUALIZADA ---
    elif part == 'media':
        try:
            print("Iniciando generación de imagen real con Vertex AI...")
            
            # 1. Preparar el prompt para la imagen
            script_text = scene.get('script', 'una imagen abstracta')
            # Mejoramos el prompt para obtener mejores resultados
            image_prompt = f"cinematic, photorealistic, high detail image for a video scene about: {script_text}"

            # 2. Obtener la resolución/aspect ratio
            # El frontend envía '16:9' o '9:16'
            aspect_ratio = data.get('config', {}).get('resolucion', '16:9')

            # 3. Generar la imagen con Imagen 2
            images = model_image.generate_images(
                prompt=image_prompt,
                number_of_images=1,
                aspect_ratio=aspect_ratio,
                # Puedes añadir un prompt negativo si quieres evitar ciertos elementos
                # negative_prompt="text, watermark, blurry"
            )

            # La API devuelve una URL temporal a la imagen generada
            temp_image_url = images[0]._image_bytes._blob.public_url
            
            # 4. Descargar la imagen de la URL temporal
            print(f"Descargando imagen desde la URL temporal de Vertex: {temp_image_url}")
            response = requests.get(temp_image_url)
            response.raise_for_status() # Lanza un error si la descarga falla
            image_bytes = response.content

            # 5. Subir la imagen a nuestro propio bucket de GCS para tener una URL permanente
            image_filename = f"image_{uuid.uuid4()}.png"
            print(f"Subiendo imagen a GCS como: {image_filename}")
            public_gcs_url = upload_to_gcs(image_bytes, image_filename, 'image/png')

            if not public_gcs_url:
                raise Exception("Fallo al subir la imagen generada a Google Cloud Storage.")

            print(f"Imagen generada y almacenada con éxito: {public_gcs_url}")
            return jsonify({"newImageUrl": public_gcs_url, "newVideoUrl": None})

        except Exception as e:
            print(f"ERROR en la generacion de media: {str(e)}")
            return jsonify({"error": f"Error al generar imagen con IA: {str(e)}"}), 500
            
    return jsonify({"error": "Parte no válida para regenerar"}), 400

@app.route('/api/generate-and-save-audio', methods=['POST'])
def generate_audio():
    data = request.get_json()
    script = data.get('script')
    voice_id = data.get('voice', 'es-US-Wavenet-B')
    if not script:
        return jsonify({"error": "El guion (script) es requerido"}), 400
    try:
        synthesis_input = texttospeech.SynthesisInput(text=script)
        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_id.split('-')[0] + '-' + voice_id.split('-')[1],
            name=voice_id
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = text_to_speech_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        file_name = f"audio_{uuid.uuid4()}.mp3"
        public_url = upload_to_gcs(response.audio_content, file_name, 'audio/mpeg')
        if public_url:
            return jsonify({"audioUrl": public_url})
        else:
            return jsonify({"error": "No se pudo subir el archivo de audio"}), 500
    except Exception as e:
        return jsonify({"error": f"Error en la API de Text-to-Speech: {str(e)}"}), 500

@app.route('/api/voice-sample', methods=['POST'])
def generate_voice_sample():
    return generate_audio()

@app.route('/api/generate-seo', methods=['POST'])
def generate_seo():
    data = request.get_json()
    guion = data.get('guion')
    nicho = data.get('nicho')
    if not guion:
        return jsonify({"error": "El guion es requerido"}), 400
    prompt = f"""
    Eres un experto en SEO para redes sociales como YouTube y TikTok.
    Basado en el siguiente guion de video para el nicho de '{nicho}', genera contenido para optimizar su alcance.
    Guion:
    ---
    {guion}
    ---
    Genera una respuesta en formato JSON con las claves "titulo", "descripcion", y "hashtags".
    El título debe ser atractivo y corto. La descripción debe ser informativa. Los hashtags deben ser una única cadena de texto separados por espacios.
    Asegúrate de que la salida sea únicamente el objeto JSON válido y nada más.
    """
    try:
        response = model_text.generate_content(prompt)
        parsed_json = safe_json_parse(response.text)
        if parsed_json:
            return jsonify(parsed_json)
        else:
            return jsonify({"error": "La respuesta del modelo de SEO no tuvo el formato esperado."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 4. EJECUCIÓN DEL SERVIDOR ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
