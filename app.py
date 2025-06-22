import os
import uuid
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import texttospeech
from google.cloud import storage

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv()

# MODIFICACIÓN PARA RENDER: Lógica para manejar credenciales de Google Cloud en producción.
# En Render, las credenciales se cargarán desde una variable de entorno segura.
# Esta función asegura que el código funcione tanto localmente como en el servidor.
if os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON'):
    # Si la variable de entorno con el contenido del JSON existe...
    credentials_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    # Guardamos este contenido en un archivo temporal que la librería de Google puede leer.
    credentials_path = '/tmp/gcp-credentials.json'
    with open(credentials_path, 'w') as f:
        f.write(credentials_json_str)
    # Le decimos a las librerías de Google que usen este archivo.
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

app = Flask(__name__)
CORS(app)

# Configuración de APIs de Google
# Esta parte no cambia, ya que las credenciales se configuran a través de las variables de entorno.
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    text_to_speech_client = texttospeech.TextToSpeechClient()
    storage_client = storage.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
except Exception as e:
    print(f"Error al configurar los clientes de Google: {e}")

model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. FUNCIONES AUXILIARES (Sin cambios) ---

def upload_to_gcs(file_stream, destination_blob_name, content_type='audio/mpeg'):
    """Sube un stream de archivo a un bucket de GCS y lo hace público."""
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
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

# --- 3. ENDPOINTS DE LA API (Sin cambios en la lógica) ---

@app.route("/")
def index():
    # Un endpoint simple para verificar que el servicio está corriendo
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
        response = model.generate_content(prompt)
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
            response = model.generate_content(prompt)
            return jsonify({"newScript": response.text.strip()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    elif part == 'media':
        print("ADVERTENCIA: La generación de imágenes no está implementada. Se devuelve un placeholder.")
        resolucion = data.get('config', {}).get('resolucion', '1920x1080')
        width, height = resolucion.split('x')
        placeholder_url = f"https://via.placeholder.com/{width}x{height}?text=IA+Fallback+Image"
        return jsonify({"newImageUrl": placeholder_url, "newVideoUrl": None})
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
        response = model.generate_content(prompt)
        parsed_json = safe_json_parse(response.text)
        if parsed_json:
            return jsonify(parsed_json)
        else:
            return jsonify({"error": "La respuesta del modelo de SEO no tuvo el formato esperado."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 4. EJECUCIÓN DEL SERVIDOR ---
# MODIFICACIÓN PARA RENDER: Esta sección ahora solo se usa para desarrollo local.
# En producción, Gunicorn ejecutará la 'app' directamente.
if __name__ == '__main__':
    # El puerto se obtiene de la variable de entorno PORT, con 5001 como default para local.
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False) # Debug se establece en False para producción
