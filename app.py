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

@retry_on_failure(retries=3, delay=2)
def _generate_and_upload_image(scene_script, aspect_ratio):
    logging.info(f"Generando imagen para el guion: '{scene_script[:50]}...' con aspect ratio: {aspect_ratio}")
    image_prompt = f"cinematic, photorealistic, high detail image for a video scene about: {scene_script}"
    
    images = model_image.generate_images(
        prompt=image_prompt,
        number_of_images=1,
        aspect_ratio=aspect_ratio,
        negative_prompt="text, watermark, logo, blurry, words, letters, signature, deformed"
    )
    
    public_gcs_url = upload_to_gcs(images[0]._image_bytes, f"images/img_{uuid.uuid4()}.png", 'image/png')
    return public_gcs_url

@retry_on_failure(retries=3, delay=2)
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


# --- 4. ENDPOINTS DE LA API ---

@app.route("/")
def index():
    return "Backend de IA para Videos v2.2 - Lógica de Narración por Prompts"

# =================================================================================
# ### Endpoint de generación inicial con CONTEXTO DE NARRACIÓN ###
# =================================================================================
@app.route('/api/generate-initial-content', methods=['POST'])
def generate_initial_content():
    try:
        data = request.get_json()
        logging.info(f"Recibida solicitud para generar contenido completo con datos: {data}")
        
        nicho = data.get('nicho', 'documentales')
        
        instruccion_veracidad = ""
        if nicho != 'biblia': 
            instruccion_veracidad = """
            - **VERACIDAD Y HECHOS REALES (REGLA CRÍTICA):** El guion DEBE basarse estrictamente en hechos y datos verificables y reales sobre el tema solicitado. Actúa como un investigador y documentalista, no como un escritor de ficción. Tu principal objetivo es informar de manera entretenida pero precisa. No inventes sucesos, fechas o detalles.
            """

        duracion_a_escenas = {"50": 4, "120": 6, "180": 8, "300": 10, "600": 15}
        numero_de_escenas = duracion_a_escenas.get(str(data.get('duracionVideo', '50')), 4)
        
        # --- CAMBIO REALIZADO: Ajuste de contexto para usar ritmo de narración ---
        prompt = f"""
        Eres un guionista experto y documentalista para videos virales de redes sociales. Tu tarea es crear un guion completo siguiendo estas instrucciones con MÁXIMA PRECISIÓN.

        **Contexto:**
        - Tema Principal: "{data.get('guionPersonalizado')}"
        - Nicho: "{nicho}"
        - Estilo de Narración Deseado: "{data.get('ritmoNarracion')}"

        **Instrucciones de Guion:**
        - **Estructura:** Genera EXACTAMENTE {numero_de_escenas} escenas.
        {instruccion_veracidad}
        - **Formato Narrativo:** Cada guion debe ser un párrafo completo, no una lista. Debe leerse como si un narrador lo estuviera contando.
        - **Texto Limpio:** El guion debe ser ÚNICAMENTE texto plano, sin etiquetas como `<speak>` o `(sonido de...)`.

        **Formato de Salida Obligatorio:**
        La respuesta DEBE SER ÚNICAMENTE un objeto JSON válido, sin explicaciones. El JSON debe tener una clave "scenes", que es un array de objetos. Cada objeto debe tener "id" y "script".
        Genera el guion ahora, siguiendo todas las reglas estrictamente.
        """

        logging.info("Enviando prompt de guion factual a Gemini.")
        response = model_text.generate_content(prompt)
        parsed_json = safe_json_parse(response.text)

        if not (parsed_json and 'scenes' in parsed_json):
            logging.error(f"La respuesta del modelo no tuvo el formato JSON esperado. Respuesta: {response.text}")
            return jsonify({"error": "La IA no pudo generar un guion con el formato correcto. Intenta de nuevo."}), 500

        logging.info(f"Guion generado con {len(parsed_json['scenes'])} escenas. Ahora generando imágenes.")
        
        aspect_ratio = data.get('resolucionVideo') or data.get('resolucion', '16:9')

        scenes_con_media = []
        for scene in parsed_json['scenes']:
            scene['id'] = scene.get('id', f'scene-{uuid.uuid4()}')
            try:
                image_url = _generate_and_upload_image(scene['script'], aspect_ratio)
                scene['imageUrl'] = image_url
                scene['videoUrl'] = None
            except Exception as e:
                logging.error(f"No se pudo generar imagen para la escena {scene['id']}: {e}", exc_info=True)
                error_img_res = '1080x1920' if aspect_ratio == '9:16' else '1920x1080'
                scene['imageUrl'] = f"https://via.placeholder.com/{error_img_res}?text=Error+al+generar+imagen"
                scene['videoUrl'] = None
            scenes_con_media.append(scene)

        logging.info("Proceso de generación unificada completado.")
        return jsonify({"scenes": scenes_con_media})
            
    except Exception as e:
        logging.error("Error inesperado en generate_initial_content.", exc_info=True)
        return jsonify({"error": f"Ocurrió un error interno al generar el contenido: {e}"}), 500

# =================================================================================
# ### Endpoint de regeneración sin cambios significativos ###
# =================================================================================
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
            logging.info(f"Regenerando guion para escena: {scene.get('id')}")
            prompt = f"Eres un guionista experto y documentalista. Reescribe el siguiente guion para una escena de video de forma creativa, concisa y **basada en hechos**. Mantén la idea central: '{scene.get('script')}'. Devuelve solo el nuevo texto del guion, sin comillas ni explicaciones."
            response = model_text.generate_content(prompt)
            new_script = response.text.strip()
            return jsonify({"newScript": new_script})
        except Exception as e:
            logging.error(f"Error al regenerar guion: {e}", exc_info=True)
            return jsonify({"error": "Error al contactar al modelo de IA para regenerar el guion."}), 500

    elif part_to_regenerate == 'media':
        try:
            logging.info(f"Regenerando media para escena: {scene.get('id')}")
            aspect_ratio = config.get('resolucion') or config.get('resolucionVideo', '16:9')
            new_image_url = _generate_and_upload_image(scene.get('script', 'una imagen abstracta'), aspect_ratio)
            return jsonify({"newImageUrl": new_image_url, "newVideoUrl": None})
        except Exception as e:
            logging.error(f"Error al regenerar media: {e}", exc_info=True)
            return jsonify({"error": f"Error al generar la nueva imagen con IA: {str(e)}"}), 500
            
    return jsonify({"error": "Parte no válida para regenerar. Debe ser 'script' o 'media'."}), 400

# =================================================================================
# ### Endpoint de audio con LÓGICA DE PROMPTS DE NARRACIÓN ###
# =================================================================================
@app.route('/api/generate-full-audio', methods=['POST'])
def generate_full_audio():
    data = request.get_json()
    plain_text_script = data.get('script')
    voice_id = data.get('voice', 'es-US-Neural2-A')
    ritmo_narracion = data.get('ritmoNarracion') # Recibimos el nuevo valor del frontend
    
    if not plain_text_script:
        return jsonify({"error": "El guion de texto es requerido"}), 400
    
    try:
        # --- INICIO DE LA NUEVA LÓGICA DE NARRACIÓN ---
        logging.info(f"Iniciando generación de SSML con el ritmo de narración: {ritmo_narracion}")

        # Definimos los prompts que nos diste
        PROMPTS_NARRACION = {
            "epico": """
            Narra este texto con un estilo épico, como si fuera parte de una película de fantasía o aventura. Usa una voz expresiva, con pausas dramáticas y un ritmo heroico que transmita emoción, tensión y grandeza. La voz puede ser masculina, femenina o neutral, lo importante es que inspire poder, determinación y aventura.
            Usa pausas largas `<break time="700ms"/>` en momentos clave y enfatiza `<emphasis level="strong">` palabras poderosas.
            """,
            "historico": """
            Narra este texto como un documental histórico. El tono debe ser sobrio, pausado y con autoridad, transmitiendo información valiosa de forma clara y confiable. Puede ser con voz masculina o femenina, pero debe sonar informativa, seria y reflexiva, como una narración que guía al espectador a través del pasado.
            Usa un ritmo ligeramente lento `<prosody rate="slow">` y pausas informativas `<break time="500ms"/>`.
            """,
            "locutor_radio": """
            Narra este texto como un locutor o locutora de radio profesional. Usa un tono dinámico, carismático y cercano al oyente. El ritmo debe ser fluido y atractivo, con ligeras variaciones para mantener la atención. La voz puede ser joven o adulta, masculina o femenina, pero siempre amigable, clara y energética.
            Usa un ritmo normal o ligeramente rápido `<prosody rate="medium">` y un tono amigable.
            """,
        }
        
        # Mapeo para ser compatible con los 'value' antiguos del HTML si aún no los has cambiado
        VALOR_MAP = {
            "es-MX": "epico",
            "es-ES": "historico",
            "en-US": "locutor_radio"
        }
        
        # Obtenemos la clave correcta ('epico', 'historico', etc.)
        ritmo_key = VALOR_MAP.get(ritmo_narracion, ritmo_narracion)
        
        # Seleccionamos la instrucción de narración. Si no se encuentra, usamos una por defecto.
        instruccion_narracion = PROMPTS_NARRACION.get(ritmo_key)
        if not instruccion_narracion:
            logging.warning(f"Ritmo de narración '{ritmo_narracion}' no reconocido. Usando instrucción por defecto.")
            instruccion_narracion = "Narra este texto de forma clara y profesional, añadiendo pausas donde sea natural."

        # Construimos el prompt final para la IA
        ssml_prompt = f"""
        Eres un director de voz experto para videos virales. Tu misión es tomar un guion de texto plano y convertirlo en una cadena de texto SSML para darle vida, usando las instrucciones específicas de narración.

        **Instrucción de Narración (Regla Maestra):**
        {instruccion_narracion}

        **Guion de Texto Plano a Convertir:**
        ---
        {plain_text_script}
        ---

        **Requisitos Técnicos:**
        1.  Usa etiquetas SSML como `<break time="..."/>`, `<emphasis level="..."/>`, y `<prosody rate="..." pitch="...">` para cumplir con la "Instrucción de Narración".
        2.  El resultado final DEBE estar envuelto en un único par de etiquetas `<speak>...</speak>`.
        3.  Devuelve ÚNICAMENTE la cadena de texto con el guion completo en formato SSML. No añadas explicaciones, comentarios, ni la palabra "ssml".
        """
        # --- FIN DE LA NUEVA LÓGICA DE NARRACIÓN ---

        response_ssml = model_text.generate_content(ssml_prompt)
        ssml_script = response_ssml.text.strip().replace("```ssml", "").replace("```", "")

        if not ssml_script.startswith('<speak>') or not ssml_script.endswith('</speak>'):
            logging.warning(f"La IA no devolvió un SSML válido. Envolviendo manualmente. Respuesta: {ssml_script[:200]}")
            ssml_script = f"<speak>{ssml_script}</speak>"
            
        public_url = _generate_audio_with_api(ssml_script, voice_id)
        logging.info(f"Audio completo generado exitosamente en {public_url}")
        
        return jsonify({"audioUrl": public_url})

    except Exception as e:
        logging.error(f"Error en generate_full_audio: {e}", exc_info=True)
        return jsonify({"error": f"No se pudo generar el audio completo: {str(e)}"}), 500

# --- Endpoints de Muestra de Voz y SEO (sin cambios) ---

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
        if "InvalidArgument" in str(e) or "does not exist" in str(e): return jsonify({"error": f"La voz seleccionada ('{voice_id}') no es válida."}), 400
        logging.error("Error al generar muestra de voz: %s", e)
        return jsonify({"error": f"No se pudo generar la muestra de voz: {str(e)}"}), 500

@app.route('/api/generate-seo', methods=['POST'])
def generate_seo():
    try:
        data = request.get_json()
        guion = data.get('guion')
        nicho = data.get('nicho')
        if not guion: return jsonify({"error": "El guion es requerido"}), 400
        prompt = f"""
        Eres un experto en SEO para redes sociales como YouTube y TikTok. Basado en el guion de video para el nicho de '{nicho}', genera un JSON con "titulo", "descripcion", y "hashtags".
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

# --- 5. EJECUCIÓN DEL SERVIDOR ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)

