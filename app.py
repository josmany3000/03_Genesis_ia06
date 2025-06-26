# Importaciones necesarias al principio de tu archivo
import os
import uuid
import json
import requests
import logging
import time
import re
import threading
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import texttospeech
from google.cloud import storage
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
# ¡NUEVA IMPORTACIÓN PARA LA SOLUCIÓN DE AUDIO!
from pydub import AudioSegment
import io

# --- 1. CONFIGURACIÓN INICIAL Y LOGGING (Sin cambios) ---
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

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

JOBS = {}

# --- Configuración de Clientes de Google (Sin cambios) ---
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    text_to_speech_client = texttospeech.TextToSpeechClient()
    storage_client = storage.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
    vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv("GCP_REGION", "us-central1"))
    logging.info("Clientes de Google (Gemini, TTS, Storage, VertexAI) configurados exitosamente.")
except Exception as e:
    logging.critical("ERROR FATAL AL CONFIGURAR CLIENTES DE GOOGLE.", exc_info=True)
    
model_text = genai.GenerativeModel('gemini-1.5-flash')
model_image = ImageGenerationModel.from_pretrained("imagegeneration@006")


# --- 2. DECORADOR DE REINTENTOS (Sin cambios) ---
def retry_on_failure(retries=3, delay=5, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if isinstance(e, IndexError):
                         logging.error(f"Error de IndexError en {func.__name__}. No se reintentará. Causa probable: contenido bloqueado por la API.")
                         raise e
                    logging.warning(f"Intento {i + 1}/{retries} para {func.__name__} falló: {e}. Reintentando en {current_delay}s...")
                    if i == retries - 1:
                        logging.error(f"Todos los {retries} intentos para {func.__name__} fallaron.", exc_info=True)
                        raise e
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator

# --- 3. FUNCIONES AUXILIARES (Sin cambios) ---
@retry_on_failure()
def upload_to_gcs(file_stream, destination_blob_name, content_type):
    logging.info(f"Iniciando subida a GCS. Bucket: {GCS_BUCKET_NAME}, Destino: {destination_blob_name}")
    if not GCS_BUCKET_NAME:
        raise ValueError("El nombre del bucket de GCS no está configurado.")
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(file_stream, content_type=content_type)
    blob.make_public()
    logging.info(f"Subida a GCS exitosa. URL pública: {blob.public_url}")
    return blob.public_url

def safe_json_parse(text):
    text = text.strip().replace('```json', '').replace('```', '')
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logging.error(f"Error al decodificar JSON. Texto problemático: {text[:500]}", exc_info=True)
        return None

@retry_on_failure()
def _get_keywords_for_image_prompt(script_text):
    prompt = f"""
    Analiza el siguiente texto de una escena de video. Extrae de 4 a 5 palabras clave (keywords) que mejor describan visualmente la escena.
    **Reglas Importantes:**
    1.  Las palabras clave deben ser seguras y neutrales, aptas para un generador de imágenes con filtros de seguridad estrictos.
    2.  Enfócate en objetos, ambientes y conceptos visuales (ej: 'nave espacial, desierto, noche, estrellas, misterio').
    3.  Evita nombres propios o acciones complejas que puedan confundir a la IA de imágenes.
    4.  Devuelve únicamente las palabras clave en español, separadas por comas. NADA MÁS.
    **Texto de la Escena:** --- {script_text} ---
    """
    response = model_text.generate_content(prompt)
    keywords = response.text.strip().replace("`", "")
    logging.info(f"Keywords generadas para el guion: '{keywords}'")
    return keywords

@retry_on_failure()
def _generate_and_upload_image(scene_script, aspect_ratio):
    keywords = _get_keywords_for_image_prompt(scene_script)
    logging.info(f"Generando imagen desde keywords: '{keywords}' con aspect ratio: {aspect_ratio}")
    image_prompt = f"cinematic, photorealistic, high detail image of: {keywords}"
    images = model_image.generate_images(
        prompt=image_prompt,
        number_of_images=1,
        aspect_ratio=aspect_ratio,
        negative_prompt="text, watermark, logo, blurry, words, letters, signature"
    )
    if not images:
        logging.warning(f"La API no devolvió imágenes para las keywords: '{keywords}'.")
        raise IndexError("La lista de imágenes generadas está vacía.")
    public_gcs_url = upload_to_gcs(images[0]._image_bytes, f"images/img_{uuid.uuid4()}.png", 'image/png')
    return public_gcs_url


# --- 4. TRABAJADOR DE FONDO PARA GENERACIÓN DE IMÁGENES ---
def _perform_image_generation(job_id, scenes, aspect_ratio):
    total_scenes = len(scenes)
    scenes_con_media = []
    try:
        for i, scene in enumerate(scenes):
            JOBS[job_id]['status'] = 'processing'
            JOBS[job_id]['progress'] = f"{i + 1}/{total_scenes}"
            logging.info(f"Trabajo {job_id}: Procesando imagen {i+1}/{total_scenes}")
            scene['id'] = scene.get('id', f'scene-{uuid.uuid4()}')
            try:
                image_url = _generate_and_upload_image(scene['script'], aspect_ratio)
                scene['imageUrl'] = image_url
                scene['videoUrl'] = None
            except Exception as e:
                logging.error(f"Trabajo {job_id}: Fallo definitivo al generar imagen para la escena {scene['id']}: {e}", exc_info=True)
                error_img_res = '1080x1920' if aspect_ratio == '9:16' else '1920x1080'
                scene['imageUrl'] = f"https://via.placeholder.com/{error_img_res}?text=Error+IA"
                scene['videoUrl'] = None
            scenes_con_media.append(scene)
            if i < total_scenes - 1:
                # ✅ RECOMENDACIÓN APLICADA: Reducido el tiempo de espera
                logging.info(f"Trabajo {job_id}: Pausando por 5 segundos para respetar la cuota de la API.")
                time.sleep(5)
        JOBS[job_id]['status'] = 'completed'
        JOBS[job_id]['result'] = {"scenes": scenes_con_media}
        logging.info(f"Trabajo {job_id} completado exitosamente.")
    except Exception as e:
        logging.error(f"Trabajo {job_id} falló catastróficamente: {e}", exc_info=True)
        JOBS[job_id]['status'] = 'error'
        JOBS[job_id]['error'] = str(e)


# --- 5. ENDPOINTS DE LA API ---
@app.route("/")
def index():
    return "Backend de IA para Videos v3.0 - Híbrido y Robusto"

@app.route('/api/generate-initial-content', methods=['POST'])
def generate_initial_content():
    try:
        data = request.get_json()
        logging.info(f"Recibida solicitud de trabajo para generar contenido con datos: {data}")
        nicho = data.get('nicho', 'documentales')
        instruccion_veracidad = ""
        if nicho != 'biblia': 
            instruccion_veracidad = "- **VERACIDAD Y HECHOS REALES (REGLA CRÍTICA):** El guion DEBE basarse estrictamente en hechos y datos verificables y reales sobre el tema solicitado."
        
        duracion_a_escenas = {"50": 4, "120": 6, "180": 8, "300": 10, "600": 15}
        numero_de_escenas = duracion_a_escenas.get(str(data.get('duracionVideo', '50')), 4)
        
        # ✅ SOLUCIÓN PROBLEMA 2 (TU PROPUESTA): Controlar la longitud del guion por escena
        palabras_totales = int(data.get('duracionVideo', 50)) * 2.5 # Aprox. 2.5 palabras por segundo para un ritmo normal
        palabras_por_escena = int(palabras_totales // numero_de_escenas)
        
        prompt = f"""
        Eres un guionista experto y documentalista. Tu tarea es crear un guion completo.
        **Instrucciones de Guion:**
        - **Idioma (REGLA INDISPENSABLE):** El guion DEBE estar escrito íntegramente en **Español Latinoamericano**.
        - **Tema Principal:** "{data.get('guionPersonalizado')}"
        - **Nicho:** "{nicho}"
        - **Estilo de Narración Deseado:** "{data.get('ritmoNarracion')}"
        - **Estructura:** Genera EXACTAMENTE {numero_de_escenas} escenas.
        {instruccion_veracidad}
        - **REGLA DE LONGITUD CRÍTICA:** Cada escena debe tener un máximo estricto de **{palabras_por_escena} palabras**. Sé conciso.
        - **Formato Narrativo:** Párrafos completos, como si un narrador lo contara.
        - **Texto Limpio:** ÚNICAMENTE texto plano, sin etiquetas.
        **Formato de Salida Obligatorio:**
        La respuesta DEBE SER ÚNICAMENTE un objeto JSON válido con una clave "scenes", que es un array de objetos. Cada objeto debe tener "id" y "script".
        """
        logging.info(f"Enviando prompt de guion a Gemini con límite de {palabras_por_escena} palabras/escena.")
        response = model_text.generate_content(prompt)
        parsed_json = safe_json_parse(response.text)
        
        if not (parsed_json and 'scenes' in parsed_json):
            logging.error(f"La IA no pudo generar un guion con el formato correcto. Respuesta: {response.text}")
            return jsonify({"error": "La IA no pudo generar un guion con el formato correcto. Intenta de nuevo."}), 500
        
        scenes = parsed_json['scenes']

        # ✅ RECOMENDACIÓN APLICADA: Validar y registrar longitud de escenas
        scenes_validadas = [s for s in scenes if s.get('script') and len(s['script'].split()) >= 5]
        if len(scenes_validadas) < len(scenes):
            logging.warning(f"Se filtraron {len(scenes) - len(scenes_validadas)} escenas por ser demasiado cortas o estar vacías.")
        
        if not scenes_validadas:
            return jsonify({"error": "La IA no generó escenas con contenido válido."}), 500
            
        scenes = scenes_validadas
        
        for i, scene in enumerate(scenes):
            word_count = len(scene['script'].split())
            logging.info(f"Escena {i+1} generada con {word_count} palabras (límite: {palabras_por_escena}).")

        logging.info(f"Guion generado con {len(scenes)} escenas. Creando trabajo de generación de imágenes.")
        aspect_ratio = data.get('resolucionVideo') or data.get('resolucion', '16:9')
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {'status': 'pending', 'progress': f'0/{len(scenes)}'}
        thread = threading.Thread(target=_perform_image_generation, args=(job_id, scenes, aspect_ratio))
        thread.start()
        return jsonify({"jobId": job_id})
    except Exception as e:
        logging.error("Error inesperado en generate_initial_content.", exc_info=True)
        return jsonify({"error": f"Ocurrió un error interno al iniciar el trabajo: {e}"}), 500

@app.route('/api/content-job-status/<job_id>', methods=['GET'])
def get_content_job_status(job_id):
    # (Sin cambios)
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Trabajo no encontrado"}), 404
    return jsonify(job)

@app.route('/api/regenerate-scene-part', methods=['POST'])
def regenerate_scene_part():
    # (Sin cambios)
    data = request.get_json()
    scene = data.get('scene')
    part_to_regenerate = data.get('part')
    config = data.get('config')
    if not all([scene, part_to_regenerate, config]):
        return jsonify({"error": "Faltan datos de escena, parte a regenerar o configuración"}), 400
    if part_to_regenerate == 'script':
        try:
            prompt = f"Eres un guionista experto. Reescribe el siguiente guion para una escena de video de forma creativa y concisa. **El nuevo guion debe estar en Español Latinoamericano.** Mantén la idea central: '{scene.get('script')}'. Devuelve solo el nuevo texto del guion, sin comillas ni explicaciones."
            response = model_text.generate_content(prompt)
            return jsonify({"newScript": response.text.strip()})
        except Exception as e:
            logging.error(f"Error al regenerar guion: {e}", exc_info=True)
            return jsonify({"error": "Error al contactar al modelo de IA para regenerar el guion."}), 500
    elif part_to_regenerate == 'media':
        try:
            aspect_ratio = config.get('resolucion') or config.get('resolucionVideo', '16:9')
            new_image_url = _generate_and_upload_image(scene.get('script', 'una imagen abstracta'), aspect_ratio)
            return jsonify({"newImageUrl": new_image_url, "newVideoUrl": None})
        except Exception as e:
            logging.error(f"Error al regenerar media: {e}", exc_info=True)
            return jsonify({"error": f"Error al generar la nueva imagen con IA: {str(e)}"}), 500
    return jsonify({"error": "Parte no válida para regenerar. Debe ser 'script' o 'media'."}), 400

# ✅ SOLUCIÓN PROBLEMA 1 (REESTRUCTURACIÓN): Generación de audio robusta
def _generate_audio_chunk_from_text(text_chunk, voice_id):
    logging.info(f"Generando audio para el trozo: '{text_chunk[:50]}...' con voz '{voice_id}'.")
    ssml_script = f"<speak>{text_chunk}<break time='800ms'/></speak>"
    
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_script)
    language_code = '-'.join(voice_id.split('-', 2)[:2])
    voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_id)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    
    @retry_on_failure(retries=3, delay=5)
    def call_tts_api():
        return text_to_speech_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
    response = call_tts_api()
    return response.audio_content

@app.route('/api/generate-full-audio', methods=['POST'])
def generate_full_audio():
    data = request.get_json()
    plain_text_script = data.get('script')
    voice_id = data.get('voice', 'es-US-Neural2-A')
    
    if not plain_text_script:
        return jsonify({"error": "El guion de texto es requerido"}), 400
        
    try:
        paragraphs = [p.strip() for p in plain_text_script.split('\n') if p.strip()]
        if not paragraphs:
             return jsonify({"error": "El guion está vacío o no tiene párrafos."}), 400

        logging.info(f"Guion dividido en {len(paragraphs)} párrafos para procesar con el método robusto.")
        
        audio_chunks = []
        for i, p in enumerate(paragraphs):
            logging.info(f"Procesando párrafo {i+1}/{len(paragraphs)}")
            try:
                audio_bytes = _generate_audio_chunk_from_text(p, voice_id)
                segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                audio_chunks.append(segment)
            except Exception as e:
                logging.error(f"Fallo al generar audio para el párrafo: '{p[:50]}...'. Error: {e}", exc_info=True)
                continue

        if not audio_chunks:
            return jsonify({"error": "No se pudo generar ningún segmento de audio."}), 500

        logging.info("Concatenando todos los segmentos de audio...")
        final_audio = sum(audio_chunks, AudioSegment.empty())
        
        final_audio_buffer = io.BytesIO()
        final_audio.export(final_audio_buffer, format="mp3")
        final_audio_buffer.seek(0)

        logging.info("Subiendo el archivo de audio final a Google Cloud Storage.")
        public_url = upload_to_gcs(
            final_audio_buffer.getvalue(), 
            f"audio/full_audio_{uuid.uuid4()}.mp3", 
            'audio/mpeg'
        )
        
        return jsonify({"audioUrl": public_url})

    except Exception as e:
        logging.error(f"Error catastrófico en generate_full_audio: {e}", exc_info=True)
        return jsonify({"error": f"No se pudo generar el audio completo: {str(e)}"}), 500

@app.route('/api/generate-seo', methods=['POST'])
def generate_seo():
    # (Sin cambios)
    try:
        data = request.get_json()
        guion = data.get('guion')
        nicho = data.get('nicho')
        if not guion: return jsonify({"error": "El guion es requerido"}), 400
        prompt = f"""
        Eres un experto en SEO para redes sociales. Basado en el guion para el nicho de '{nicho}', genera un JSON con "titulo", "descripcion", y "hashtags". **Todo el contenido SEO debe estar en Español.**
        Guion: --- {guion} ---
        Asegúrate de que la salida sea únicamente el objeto JSON válido.
        """
        response = model_text.generate_content(prompt)
        parsed_json = safe_json_parse(response.text)
        if parsed_json: return jsonify(parsed_json)
        else: return jsonify({"error": "La respuesta del modelo de SEO no tuvo el formato esperado."}), 500
    except Exception as e:
        logging.error("Error al generar SEO: %s", e)
        return jsonify({"error": "Ocurrió un error interno al generar el contenido SEO."}), 500

@app.route('/api/voice-sample', methods=['POST'])
def generate_voice_sample():
    # Se modifica para usar la nueva función de chunk, aunque no es estrictamente necesario
    data = request.get_json()
    voice_id = data.get('voice')
    if not voice_id: return jsonify({"error": "Se requiere un ID de voz"}), 400
    try:
        sample_text = "Hola, esta es una prueba de la voz seleccionada para la narración."
        audio_content = _generate_audio_chunk_from_text(sample_text, voice_id)
        public_url = upload_to_gcs(audio_content, f"audio/sample_{uuid.uuid4()}.mp3", 'audio/mpeg')
        return jsonify({"audioUrl": public_url})
    except Exception as e:
        logging.error("Error al generar muestra de voz: %s", e)
        return jsonify({"error": f"No se pudo generar la muestra de voz: {str(e)}"}), 500

# --- 6. EJECUCIÓN DEL SERVIDOR (Sin cambios) ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
    
