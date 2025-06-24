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
    logging.info(f"Iniciando subida a GCS. Bucket: {GCS_BUCKET_NAME}, Destino: {destination_blob_name}")
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

@retry_on_failure(retries=3, delay=2, backoff=2)
def _generate_audio_with_api(ssml_script, voice_id):
    """Función interna que procesa SSML para generar audio."""
    logging.info(f"Llamando a la API de Google TTS con SSML y voz '{voice_id}'.")
    
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_script)
    
    language_code = '-'.join(voice_id.split('-', 2)[:2])
    voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_id)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    
    response = text_to_speech_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    logging.info("Respuesta de la API de TTS recibida.")

    file_name = f"audio_{uuid.uuid4()}.mp3"
    public_url = upload_to_gcs(response.audio_content, file_name, 'audio/mpeg')
    
    return public_url


# --- 4. ENDPOINTS DE LA API ---

@app.route("/")
def index():
    return "Backend de IA para Videos - ¡Corriendo con lógica de audio unificado y SSML dinámico!"

# =================================================================================
# ### CAMBIO 1: El guion inicial ahora se genera como TEXTO PLANO ###
# =================================================================================
@app.route('/api/generate-initial-content', methods=['POST'])
def generate_initial_content():
    try:
        data = request.get_json()
        logging.info(f"Recibida solicitud para generar guion con datos: {data}")

        duracion_a_escenas = {"50": 4, "120": 6, "180": 8, "300": 10, "600": 15}
        duracion_seleccionada = str(data.get('duracionVideo', '50'))
        numero_de_escenas = duracion_a_escenas.get(duracion_seleccionada, 4)
        
        cantidad_ganchos = int(data.get('cantidadGanchos', 0))
        usar_tendencias = data.get('usarTendencias', False)
        nicho = data.get('nicho', 'tecnologia')
        
        instruccion_ganchos = ""
        if cantidad_ganchos > 0:
            instruccion_ganchos = f"""- Ganchos Virales: Antes de las escenas principales, genera EXACTAMENTE {cantidad_ganchos} escenas 'gancho' muy cortas e impactantes. Su 'id' debe ser "hook-1", "hook-2", etc."""

        instruccion_tendencias = ""
        if usar_tendencias:
            instruccion_tendencias = f"""- Relevancia Cultural: Incorpora temas o formatos de tendencia actual en redes sociales para el nicho de '{nicho}'."""

        prompt = f"""
        Eres un guionista experto para videos virales de redes sociales. Tu tarea es crear un guion completo siguiendo estas instrucciones.

        **Contexto del Video:**
        - Tema Principal: "{data.get('guionPersonalizado')}"
        - Nicho: "{nicho}"
        - Idioma del Guion: "{data.get('idioma')}"

        **Requisitos de Estructura (MUY IMPORTANTE):**
        - Número de Escenas Principales: Debes generar EXACTAMENTE {numero_de_escenas} escenas principales.
        {instruccion_ganchos}
        {instruccion_tendencias}
        - ¡Importante! El guion para cada escena debe ser **texto plano, claro y conciso**. No incluyas ninguna etiqueta como <speak> o <prosody>.

        **Formato de Salida Obligatorio:**
        La respuesta DEBE SER ÚNICAMENTE un objeto JSON válido. El JSON debe tener una única clave raíz "scenes". El valor de "scenes" es un array de objetos. Cada objeto debe contener:
        1. "id": Un identificador único (ej: "hook-1", "scene-1").
        2. "script": El guion de la escena como una cadena de **TEXTO PLANO**.

        Genera el guion ahora.
        """

        logging.info("Enviando prompt de TEXTO PLANO a Gemini.")
        response = model_text.generate_content(prompt)
        parsed_json = safe_json_parse(response.text)

        if parsed_json and 'scenes' in parsed_json:
            for i, scene in enumerate(parsed_json['scenes']):
                scene['id'] = scene.get('id', f'scene-{uuid.uuid4()}')
            logging.info(f"Guion de texto plano generado con {len(parsed_json['scenes'])} escenas.")
            return jsonify(parsed_json)
        else:
            logging.error(f"La respuesta del modelo no tuvo el formato JSON esperado. Respuesta: {response.text}")
            return jsonify({"error": "La IA no pudo generar un guion con el formato correcto. Intenta de nuevo."}), 500
            
    except Exception as e:
        logging.error("Error inesperado en generate_initial_content.", exc_info=True)
        return jsonify({"error": "Ocurrió un error interno al generar el guion."}), 500


@app.route('/api/regenerate-scene-part', methods=['POST'])
def regenerate_scene_part():
    data = request.get_json()
    scene = data.get('scene')
    part = data.get('part')

    if not scene or not part:
        return jsonify({"error": "Faltan datos de escena o parte a regenerar"}), 400

    if part == 'script':
        try:
            prompt = f"Eres un guionista. Reescribe el siguiente guion para una escena de video de forma creativa, manteniendo la idea central: '{scene.get('script')}'. Devuelve solo el texto del nuevo guion, sin comillas ni texto adicional."
            response = model_text.generate_content(prompt)
            # ### CAMBIO: Ahora devuelve texto plano, como debe ser.
            return jsonify({"newScript": response.text.strip()})
        except Exception as e:
            logging.error("Error al regenerar guion: %s", e)
            return jsonify({"error": "Error al contactar al modelo de IA."}), 500

    elif part == 'media':
        try:
            script_text = scene.get('script', 'una imagen abstracta')
            image_prompt = f"cinematic, photorealistic, high detail image for a video scene about: {script_text}"
            aspect_ratio = data.get('config', {}).get('resolucion', '16:9')
            images = model_image.generate_images(
                prompt=image_prompt, number_of_images=1, aspect_ratio=aspect_ratio,
                negative_prompt="text, watermark, blurry, words, letters"
            )
            public_gcs_url = upload_to_gcs(images[0]._image_bytes, f"image_{uuid.uuid4()}.png", 'image/png')
            return jsonify({"newImageUrl": public_gcs_url, "newVideoUrl": None})
        except Exception as e:
            logging.error("Error al regenerar media: %s", e)
            return jsonify({"error": f"Error al generar imagen con IA: {str(e)}"}), 500
            
    return jsonify({"error": "Parte no válida para regenerar"}), 400

# =================================================================================
# ### CAMBIO 2: Endpoint de audio ahora es inteligente. Convierte texto plano a SSML ###
# =================================================================================
@app.route('/api/generate-full-audio', methods=['POST'])
def generate_full_audio():
    data = request.get_json()
    plain_text_script = data.get('script')
    nicho = data.get('nicho', 'tecnologia')
    voice_id = data.get('voice', 'es-US-Neural2-A')
    
    if not plain_text_script:
        return jsonify({"error": "El guion de texto es requerido"}), 400
    
    try:
        logging.info(f"Iniciando conversión de texto plano a SSML para el nicho: {nicho}")

        # Prompt para que la IA convierta el texto plano en un guion con dirección de voz (SSML)
        ssml_prompt = f"""
        Eres un director de voz experto. Tu tarea es tomar un guion de texto plano y enriquecerlo con etiquetas SSML para darle vida, ritmo y el tono adecuado para un video viral.

        **Instrucciones:**
        - **Nicho del Video:** {nicho}
        - **Guion de Texto Plano:**
        ---
        {plain_text_script}
        ---

        **Tu Misión:**
        1.  Analiza el guion y el nicho.
        2.  Inserta etiquetas SSML de forma creativa y profesional. Usa `<break time="..."/>` para pausas dramáticas o para separar ideas, `<emphasis level="..."/>` para resaltar palabras clave, y `<prosody rate="..." pitch="...">` para controlar el ritmo y el tono.
            - **Misterio/Terror:** Usa pausas largas (ej. `<break time="700ms"/>`), ritmo lento (`rate="slow"`) y tono grave (`pitch="-15%"`).
            - **Finanzas/Tecnología:** Usa un ritmo claro, seguro (`rate="medium"`) y enfatiza (`emphasis`) conceptos importantes.
            - **Documental/Historia:** Usa un tono narrativo, calmado (`rate="slow"`) y respetuoso, con pausas claras entre párrafos.
        3.  Asegúrate de que todo el resultado final esté envuelto en un único par de etiquetas `<speak>...</speak>`.

        **Formato de Salida Obligatorio:**
        Devuelve **ÚNICAMENTE** la cadena de texto con el guion completo en formato SSML. No añadas explicaciones ni texto adicional.
        """
        
        response_ssml = model_text.generate_content(ssml_prompt)
        ssml_script = response_ssml.text.strip().replace("```ssml", "").replace("```", "")

        # Validar que la respuesta sea un SSML básico
        if not ssml_script.startswith('<speak>') or not ssml_script.endswith('</speak>'):
            logging.warning(f"La IA no devolvió un SSML válido. Envolviendo manualmente. Respuesta: {ssml_script[:200]}")
            ssml_script = f"<speak>{ssml_script}</speak>"
            
        logging.info("SSML generado exitosamente. Enviando a la API de TTS.")
        
        public_url = _generate_audio_with_api(ssml_script, voice_id)
        logging.info(f"Audio completo generado exitosamente en {public_url}")
        
        return jsonify({"audioUrl": public_url})

    except Exception as e:
        logging.error(f"Error en generate_full_audio: {e}", exc_info=True)
        return jsonify({"error": f"No se pudo generar el audio completo: {str(e)}"}), 500


# ### CAMBIO 3: Endpoint obsoleto eliminado y voice-sample simplificado ###
@app.route('/api/voice-sample', methods=['POST'])
def generate_voice_sample():
    data = request.get_json()
    voice_id = data.get('voice')
    if not voice_id:
        return jsonify({"error": "Se requiere un ID de voz"}), 400
    try:
        # Usa un SSML simple y predefinido para la muestra
        sample_ssml = "<speak>Hola, esta es una prueba de la voz seleccionada para la narración.</speak>"
        public_url = _generate_audio_with_api(sample_ssml, voice_id)
        return jsonify({"audioUrl": public_url})
    except Exception as e:
        if "InvalidArgument" in str(e) or "does not exist" in str(e):
             return jsonify({"error": f"La voz seleccionada ('{voice_id}') no es válida."}), 400
        logging.error("Error al generar muestra de voz: %s", e)
        return jsonify({"error": f"No se pudo generar la muestra de voz: {str(e)}"}), 500

@app.route('/api/generate-seo', methods=['POST'])
def generate_seo():
    try:
        data = request.get_json()
        guion = data.get('guion')
        nicho = data.get('nicho')
        if not guion:
            return jsonify({"error": "El guion es requerido"}), 400
        prompt = f"""
        Eres un experto en SEO para redes sociales como YouTube y TikTok. Basado en el guion de video para el nicho de '{nicho}', genera un JSON con "titulo", "descripcion", y "hashtags".
        Guion: --- {guion} ---
        Asegúrate de que la salida sea únicamente el objeto JSON válido.
        """
        response = model_text.generate_content(prompt)
        parsed_json = safe_json_parse(response.text)
        if parsed_json:
            return jsonify(parsed_json)
        else:
            return jsonify({"error": "La respuesta del modelo de SEO no tuvo el formato esperado."}), 500
    except Exception as e:
        logging.error("Error al generar SEO: %s", e)
        return jsonify({"error": "Ocurrió un error interno al generar el contenido SEO."}), 500

# --- 5. EJECUCIÓN DEL SERVIDOR ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
    
