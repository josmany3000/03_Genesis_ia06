import os
import uuid
import json
import requests
import logging
import time
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import texttospeech
from google.cloud import storage
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# --- 1. CONFIGURACIÓN INICIAL Y LOGGING ---
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# Carga de credenciales desde variable de entorno si existe
if os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON'):
    credentials_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    credentials_path = f'/tmp/{uuid.uuid4()}_gcp-credentials.json'
    try:
        with open(credentials_path, 'w') as f:
            f.write(credentials_json_str)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        logging.info("Credenciales de GCP cargadas desde variable de entorno.")
    except Exception as e:
        logging.error("No se pudieron escribir las credenciales de GCP en el archivo temporal.", exc_info=True)

app = Flask(__name__)
CORS(app)

# --- Configuración de Clientes de Google ---
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    text_to_speech_client = texttospeech.TextToSpeechClient()
    storage_client = storage.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
    vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv("GCP_REGION", "us-central1"))
    logging.info("Clientes de Google (Gemini, TTS, Storage, VertexAI) configurados exitosamente.")
except Exception as e:
    logging.critical("ERROR FATAL AL CONFIGURAR CLIENTES DE GOOGLE.", exc_info=True)
    # Si los clientes no se inician, no tiene sentido continuar.
    # En un entorno de producción, esto debería detener la aplicación.
    
model_text = genai.GenerativeModel('gemini-1.5-flash')
model_image = ImageGenerationModel.from_pretrained("imagegeneration@006")


# --- 2. DECORADOR DE REINTENTOS ---
def retry_on_failure(retries=3, delay=2, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.warning(f"Intento {i + 1}/{retries} para {func.__name__} falló: {e}. Reintentando en {current_delay}s...")
                    if i == retries - 1:
                        logging.error(f"Todos los {retries} intentos para {func.__name__} fallaron.", exc_info=True)
                        raise e
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator

# --- 3. FUNCIONES AUXILIARES ---

@retry_on_failure(retries=3, delay=2)
def upload_to_gcs(file_stream, destination_blob_name, content_type='audio/mpeg'):
    """Sube un stream de archivo a un bucket de GCS y lo hace público."""
    logging.info(f"Iniciando subida a GCS. Bucket: {GCS_BUCKET_NAME}, Destino: {destination_blob_name}")
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    
    # upload_from_string espera bytes
    blob.upload_from_string(file_stream, content_type=content_type)
    
    blob.make_public()
    logging.info(f"Subida a GCS exitosa. URL pública: {blob.public_url}")
    return blob.public_url

def safe_json_parse(text):
    """Intenta parsear un string JSON, limpiando el formato si es necesario."""
    text = text.strip().replace('```json', '').replace('```', '')
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logging.error(f"Error al decodificar JSON. Texto problemático: {text[:500]}", exc_info=True)
        return None

@retry_on_failure(retries=3, delay=2, backoff=2)
def _generate_audio_with_api(script, voice_id):
    """Función interna que maneja la lógica de API para generar y subir audio."""
    logging.info(f"Llamando a la API de Google TTS con voz '{voice_id}'.")
    synthesis_input = texttospeech.SynthesisInput(text=script)
    
    # --- MEJORA ---
    # Forma más segura de obtener el código de lenguaje.
    # Divide el string por el guion un máximo de 2 veces y une las primeras dos partes.
    # Ej: 'es-US-Wavenet-A' -> ['es', 'US', 'Wavenet-A'] -> ['es', 'US'] -> 'es-US'
    language_code = '-'.join(voice_id.split('-', 2)[:2])
    
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_id
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    
    response = text_to_speech_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    logging.info("Respuesta de la API de TTS recibida.")

    file_name = f"audio_{uuid.uuid4()}.mp3"
    public_url = upload_to_gcs(response.audio_content, file_name, 'audio/mpeg')
    
    return public_url


# --- 4. ENDPOINTS DE LA API ---

@app.route("/")
def index():
    return "Backend de IA para Videos - ¡Corriendo con reintentos y lógica mejorada!"

@app.route('/api/generate-initial-content', methods=['POST'])
def generate_initial_content():
    try:
        data = request.get_json()
        logging.info(f"Recibida solicitud para generar guion inicial con tema: '{data.get('guionPersonalizado')}'")
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
        response = model_text.generate_content(prompt)
        parsed_json = safe_json_parse(response.text)
        if parsed_json and 'scenes' in parsed_json:
            for i, scene in enumerate(parsed_json['scenes']):
                scene['id'] = scene.get('id', f'scene-{uuid.uuid4()}')
            logging.info("Guion inicial generado y parseado correctamente.")
            return jsonify(parsed_json)
        else:
            logging.error("La respuesta del modelo de guion no tuvo el formato JSON esperado.")
            return jsonify({"error": "La respuesta del modelo no tuvo el formato esperado."}), 500
    except Exception as e:
        logging.error("Error inesperado en generate_initial_content.", exc_info=True)
        return jsonify({"error": "Ocurrió un error interno al generar el guion."}), 500

@app.route('/api/regenerate-scene-part', methods=['POST'])
def regenerate_scene_part():
    data = request.get_json()
    scene = data.get('scene')
    part = data.get('part') # 'script' o 'media'
    
    if not scene or not part:
        return jsonify({"error": "Faltan datos de escena o parte a regenerar"}), 400

    if part == 'script':
        try:
            logging.info(f"Regenerando SCRIPT para escena ID: {scene.get('id')}")
            prompt = f"Eres un guionista. Reescribe el siguiente guion para una escena de video de forma creativa, manteniendo la idea central. Guion original: '{scene.get('script')}'. Devuelve solo el texto del nuevo guion, sin comillas ni texto introductorio."
            response = model_text.generate_content(prompt)
            new_script = response.text.strip()
            return jsonify({"newScript": new_script})
        except Exception as e:
            logging.error(f"Error regenerando SCRIPT para escena ID: {scene.get('id')}.", exc_info=True)
            return jsonify({"error": "Error al contactar al modelo de IA."}), 500

    elif part == 'media':
        try:
            logging.info(f"Regenerando MEDIA para escena ID: {scene.get('id')}")
            script_text = scene.get('script', 'una imagen abstracta')
            image_prompt = f"cinematic, photorealistic, high detail image for a video scene about: {script_text}"
            aspect_ratio = data.get('config', {}).get('resolucion', '16:9')

            images = model_image.generate_images(
                prompt=image_prompt,
                number_of_images=1,
                aspect_ratio=aspect_ratio,
                negative_prompt="text, watermark, blurry, words, letters"
            )

            # Para Vertex AI, es más seguro obtener los bytes directamente que depender de una URL temporal
            image_bytes = images[0].image_bytes
            image_filename = f"image_{uuid.uuid4()}.png"
            public_gcs_url = upload_to_gcs(image_bytes, image_filename, 'image/png')
            
            return jsonify({"newImageUrl": public_gcs_url, "newVideoUrl": None})
        except Exception as e:
            logging.error(f"Error regenerando MEDIA para escena ID: {scene.get('id')}.", exc_info=True)
            return jsonify({"error": f"Error al generar imagen con IA: {str(e)}"}), 500

    return jsonify({"error": "Parte no válida para regenerar"}), 400

@app.route('/api/generate-and-save-audio', methods=['POST'])
def generate_audio():
    data = request.get_json()
    script = data.get('script')
    # --- CORRECCIÓN ---
    # Usamos una voz por defecto que sabemos que es válida ('es-US-Wavenet-A').
    # El error original 'es-US-Wavenet-D' provenía del valor que enviaba el frontend.
    voice_id = data.get('voice', 'es-US-Wavenet-A')

    if not script:
        return jsonify({"error": "El guion (script) es requerido"}), 400
    
    logging.info(f"Solicitud para generar audio con voz '{voice_id}' para script: '{script[:60]}...'")
    
    try:
        public_url = _generate_audio_with_api(script, voice_id)
        logging.info(f"Proceso de generación de audio completado. URL: {public_url}")
        return jsonify({"audioUrl": public_url})
    except Exception as e:
        logging.error("FALLO CRÍTICO en la generación de audio después de reintentos.", exc_info=True)
        # Devolvemos el mensaje de error específico de la API si es un error de argumento inválido
        if "InvalidArgument" in str(e) or "does not exist" in str(e):
             return jsonify({"error": f"La voz seleccionada ('{voice_id}') no es válida. Por favor, elige otra."}), 400
        return jsonify({"error": f"No se pudo generar el audio: {str(e)}"}), 500

@app.route('/api/voice-sample', methods=['POST'])
def generate_voice_sample():
    logging.info("Recibida solicitud de muestra de voz.")
    # Reutilizamos la lógica de generate_audio
    return generate_audio()

@app.route('/api/generate-seo', methods=['POST'])
def generate_seo():
    try:
        data = request.get_json()
        guion = data.get('guion')
        nicho = data.get('nicho')
        
        if not guion:
            return jsonify({"error": "El guion es requerido"}), 400
            
        logging.info(f"Generando SEO para nicho: {nicho}")
        prompt = f"""
        Eres un experto en SEO para redes sociales como YouTube y TikTok.
        Basado en el siguiente guion de video para el nicho de '{nicho}', genera contenido para optimizar su alcance.
        Guion:
        ---
        {guion}
        ---
        Genera una respuesta en formato JSON con las claves "titulo", "descripcion", y "hashtags".
        El título debe ser atractivo y corto. La descripción debe ser informativa. Los hashtags deben ser una única cadena de texto separados por espacios y comenzando con #.
        Asegúrate de que la salida sea únicamente el objeto JSON válido y nada más.
        """
        response = model_text.generate_content(prompt)
        parsed_json = safe_json_parse(response.text)
        
        if parsed_json:
            return jsonify(parsed_json)
        else:
            return jsonify({"error": "La respuesta del modelo de SEO no tuvo el formato esperado."}), 500
    except Exception as e:
        logging.error("Error inesperado en generate_seo.", exc_info=True)
        return jsonify({"error": "Ocurrió un error interno al generar el contenido SEO."}), 500

# --- 5. EJECUCIÓN DEL SERVIDOR ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
        
