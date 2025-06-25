import os
import uuid
import json
import requests
import logging
import time
import re # Importamos la librería de expresiones regulares para la limpieza
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

# --- 1. CONFIGURACIÓN INICIAL Y LOGGING ---
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
    
model_text = genai.GenerativeModel('gemini-1.5-flash')
model_image = ImageGenerationModel.from_pretrained("imagegeneration@006")


# --- 2. DECORADOR DE REINTENTOS ---
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

# --- 3. FUNCIONES AUXILIARES ---
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
    """Usa la IA de texto para extraer palabras clave seguras y visuales del guion."""
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

@retry_on_failure()
def _generate_audio_with_api(ssml_script, voice_id):
    logging.info(f"Llamando a la API de Google TTS con SSML y voz '{voice_id}'.")
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_script)
    language_code = '-'.join(voice_id.split('-', 2)[:2])
    voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_id)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = text_to_speech_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    logging.info("Respuesta de la API de TTS recibida.")
    public_url = upload_to_gcs(response.audio_content, f"audio/audio_{uuid.uuid4()}.mp3", 'audio/mpeg')
    return public_url


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
                logging.info(f"Trabajo {job_id}: Pausando por 10 segundos para respetar la cuota de la API.")
                time.sleep(10)
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
    return "Backend de IA para Videos v2.8 - Limpieza SSML Avanzada"

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
        prompt = f"""
        Eres un guionista experto y documentalista. Tu tarea es crear un guion completo.
        **Instrucciones de Guion:**
        - **Idioma (REGLA INDISPENSABLE):** El guion DEBE estar escrito íntegramente en **Español Latinoamericano**. Utiliza un lenguaje natural y claro para esa región.
        - **Tema Principal:** "{data.get('guionPersonalizado')}"
        - **Nicho:** "{nicho}"
        - **Estilo de Narración Deseado:** "{data.get('ritmoNarracion')}"
        - **Estructura:** Genera EXACTAMENTE {numero_de_escenas} escenas.
        {instruccion_veracidad}
        - **Formato Narrativo:** Párrafos completos, como si un narrador lo contara.
        - **Texto Limpio:** ÚNICAMENTE texto plano, sin etiquetas.
        **Formato de Salida Obligatorio:**
        La respuesta DEBE SER ÚNICAMENTE un objeto JSON válido con una clave "scenes", que es un array de objetos. Cada objeto debe tener "id" y "script".
        """
        logging.info("Enviando prompt de guion a Gemini.")
        response = model_text.generate_content(prompt)
        parsed_json = safe_json_parse(response.text)
        if not (parsed_json and 'scenes' in parsed_json):
            logging.error(f"La IA no pudo generar un guion con el formato correcto. Respuesta: {response.text}")
            return jsonify({"error": "La IA no pudo generar un guion con el formato correcto. Intenta de nuevo."}), 500
        scenes = parsed_json['scenes']
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
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Trabajo no encontrado"}), 404
    return jsonify(job)

@app.route('/api/regenerate-scene-part', methods=['POST'])
def regenerate_scene_part():
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

# --- INICIO DE LA SOLUCIÓN: Limpieza de Audio ---
@app.route('/api/generate-full-audio', methods=['POST'])
def generate_full_audio():
    data = request.get_json()
    plain_text_script = data.get('script')
    voice_id = data.get('voice', 'es-US-Neural2-A')
    ritmo_narracion = data.get('ritmoNarracion')
    
    if not plain_text_script:
        return jsonify({"error": "El guion de texto es requerido"}), 400
    
    try:
        logging.info(f"Iniciando generación de SSML con el ritmo de narración: {ritmo_narracion}")
        PROMPTS_NARRACION = {
            "epico": "Narra este texto con un estilo épico... Usa pausas largas y enfatiza palabras poderosas.",
            "historico": "Narra este texto como un documental histórico... Usa un ritmo ligeramente lento y pausas informativas.",
            "locutor_radio": "Narra este texto como un locutor de radio profesional... Usa un ritmo normal o ligeramente rápido y un tono amigable.",
        }
        VALOR_MAP = {"es-MX": "epico", "es-ES": "historico", "en-US": "locutor_radio"}
        ritmo_key = VALOR_MAP.get(ritmo_narracion, ritmo_narracion)
        instruccion_narracion = PROMPTS_NARRACION.get(ritmo_key, "Narra este texto de forma clara y profesional.")

        ssml_prompt = f"""
        Eres un director de voz experto. Tu misión es tomar un guion y convertirlo en SSML.
        **Instrucción de Narración (Regla Maestra):** {instruccion_narracion}
        **Guion de Texto Plano a Convertir:** --- {plain_text_script} ---
        **Requisitos Técnicos:** Usa etiquetas SSML. El resultado debe estar envuelto en `<speak>...</speak>`. Devuelve ÚNICAMENTE la cadena SSML. No incluyas `<?xml...?>` ni ` ``` `
        """
        
        response_ssml = model_text.generate_content(ssml_prompt)
        raw_ssml = response_ssml.text

        logging.info(f"Respuesta cruda de SSML de la IA: {raw_ssml[:200]}") # Log para depuración

        # Limpieza robusta de SSML: extrae solo el contenido dentro de <speak>
        match = re.search(r'<speak>(.*)</speak>', raw_ssml, re.DOTALL | re.IGNORECASE)
        if match:
            inner_content = match.group(1).strip()
            ssml_script = f"<speak>{inner_content}</speak>"
            logging.info("SSML extraído y limpiado exitosamente.")
        else:
            # Si no encuentra <speak>, asume que toda la respuesta es el guion
            # y lo envuelve manualmente, eliminando cualquier posible basura.
            logging.warning("No se encontraron etiquetas <speak>. Envolviendo texto crudo.")
            cleaned_text = raw_ssml.replace("```ssml", "").replace("```", "").strip()
            ssml_script = f"<speak>{cleaned_text}</speak>"

        public_url = _generate_audio_with_api(ssml_script, voice_id)
        return jsonify({"audioUrl": public_url})

    except Exception as e:
        logging.error(f"Error en generate_full_audio: {e}", exc_info=True)
        return jsonify({"error": f"No se pudo generar el audio completo: {str(e)}"}), 500
# --- FIN DE LA SOLUCIÓN ---


@app.route('/api/generate-seo', methods=['POST'])
def generate_seo():
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
    data = request.get_json()
    voice_id = data.get('voice')
    if not voice_id: return jsonify({"error": "Se requiere un ID de voz"}), 400
    try:
        sample_ssml = "<speak>Hola, esta es una prueba de la voz seleccionada para la narración.</speak>"
        public_url = _generate_audio_with_api(sample_ssml, voice_id)
        return jsonify({"audioUrl": public_url})
    except Exception as e:
        logging.error("Error al generar muestra de voz: %s", e)
        return jsonify({"error": f"No se pudo generar la muestra de voz: {str(e)}"}), 500

# --- 6. EJECUCIÓN DEL SERVIDOR ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
            
